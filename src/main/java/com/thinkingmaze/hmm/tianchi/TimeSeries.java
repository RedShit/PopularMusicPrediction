package com.thinkingmaze.hmm.tianchi;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import com.thinkingmaze.hmm.GLRHMM;
import com.thinkingmaze.hmm.HMM;
import com.thinkingmaze.hmm.eval.EvalGLRHMM;
import com.thinkingmaze.hmm.observation.Observation;
import com.thinkingmaze.hmm.observation.Observations;
import com.thinkingmaze.hmm.train.TrainGLRHMM;
import com.thinkingmaze.hmm.train.Training;

public class TimeSeries {
	private String possionFilePath;
	private String trainFilePath;
	private String testFilePath;
	private String predictFilePath;
	private int obsNumber;
	private int obsLength;
	private final int deltaLength;
	private int[][] real;
	private double[] lamdas;
	private Random rng;
	
	public TimeSeries(String possionFilePath, String trainFilePath, String testFilePath, 
			String predictFilePath, int obsNumber, int obsLength) throws FileNotFoundException {
		this.possionFilePath = possionFilePath;
		this.trainFilePath = trainFilePath;
		this.testFilePath = testFilePath;
		this.predictFilePath = predictFilePath;
		this.obsNumber = obsNumber;
		this.obsLength = obsLength;
		this.deltaLength = setLamda();
		this.rng = new Random(2234);
	}
	
	private int setLamda() throws FileNotFoundException {
		// TODO Auto-generated method stub
		Scanner sin = new Scanner(new File(possionFilePath));
		List<Double> tempLamda = new ArrayList<Double>();
		while(sin.hasNextLine()){
			String[] line = sin.nextLine().split(",");
			for(String str : line){
				tempLamda.add(Double.parseDouble(str));
			}
		}
		this.lamdas = new double[tempLamda.size()];
		for(int i = 0; i < tempLamda.size(); i++)
			this.lamdas[i] = tempLamda.get(i);
		sin.close();
		return this.lamdas.length;
	}

	public void run() throws IOException {
		Observations obs = getObservation(trainFilePath, obsNumber, obsLength, false);
		crossValidate(obs, 10, this.deltaLength);
	}

	/**
	 * Reads into the text file and creates an instance of Observations
	 * containing the time series.
	 * <p>
	 * The file name must specify the type of file(e.g. name.txt)
	 * 
	 * @param filename
	 *            The name of the text file
	 * @param obsNumber
	 *            The number of observations
	 * @param obsLength
	 *            The length of time series
	 * @return An instance of Observations containing the time series
	 * 
	 */
	public Observations getObservation(String filename, int obsNumber, int obsLength, boolean show) {

		int[][] result = new int[obsNumber][obsLength];
		this.real = new int[obsNumber][obsLength];
		try {
			File file = new File(filename);
			Scanner scan = new Scanner(file);
			if(filename.endsWith(".csv")){
				for(int i = 0; i < result.length; i++){
					String[] lines = scan.nextLine().split(",");
					for (int j = 0; j < result[0].length; j++){
						real[i][j] = Integer.parseInt(lines[j+1]);
						int temp = getDelta(real[i][j]);
						if(show) System.out.print(temp + " ");
						result[i][j] = temp;
					}
					if(show) System.out.println();
				}
			}
			else{
				while (scan.hasNext()) {
					for (int i = 0; i < result.length; i++) {
						for (int j = 0; j < result[0].length; j++) {
							String line = scan.next();
							int temp = Integer.parseInt(line);
	
							result[i][j] = temp;
						}
					}
				}
			}
			scan.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

		Observations observations = new Observations();

		for (int i = 0; i < obsNumber; i++) {
			int[] obv = new int[obsLength];

			for (int j = 0; j < obsLength; j++) {
				obv[j] = result[i][j];
			}
			
			Observation observation = new Observation(obv);
			observations.add(observation);
			
		}

		return observations;
	}

	private int getDelta(int d) {
		// TODO Auto-generated method stub
		double[] p = new double[this.lamdas.length];
		double sum = 0;
		for(int i = 0; i < this.lamdas.length; i++){
			p[i] = getPoisson(this.lamdas[i],d);
			sum += p[i];
		}
		if(sum <= 0.0)
			throw new NullPointerException("getDelta: sum <= 0.0.");
		double r = rng.nextDouble();
		for(int i = 0; i < this.lamdas.length; i++){
			if(r <= p[i]/sum) return i+1;
			r -= p[i]/sum;
		}
		return 1;
	}

	private double getPoisson(double lamda, int k){
		double res = 1.0;
		for(int i = 1, j = 1; i <= k || j <= lamda;){
			if(res > 1e10 && j <= lamda){
				res /= Math.E;
				j++;
				continue;
			}
			else if(i <= k){
				res = res*lamda/i;
				i++;
			}
			else if(j <= lamda){
				res /= Math.E;
				j++;
			}
			//System.out.println(res + " " + j);
		}
		return res = res/Math.exp(lamda-(int)(lamda));
	}
	
	
	
	/**
	 * Reads into an instance of Observations and performs a cross-validation by
	 * training on each of the data.
	 * <p>
	 * General Left-to-Right topology is used, since time series is being dealt
	 * 
	 * @param obs
	 *            The instance of Observations that will be cross-validated
	 * @param stateLength
	 *            Number of states. Larger number will result in longer training
	 *            time.
	 * @param delta
	 *            Number of types of observations. In this case, 3.
	 * @throws IOException 
	 */

	public void crossValidate( Observations obs, 
			int stateLength, int deltaLength ) throws IOException {
		Scanner sin = new Scanner(new File(testFilePath));
		FileWriter predictFile = new FileWriter(new File(predictFilePath));
		HMM timeSeriesHMM = new GLRHMM("WeatherData", stateLength, deltaLength, obs.getSize());
		Training trainGLRHMM = new TrainGLRHMM(obs, timeSeriesHMM);
		System.out.println("Initial Evaluation Value: " + trainGLRHMM.forward());
		System.out.println();
		int epoch = 200;
		for (int e = 0; e < epoch; e++) {

			System.out.println("Training Observations #" + (e + 1));
			trainGLRHMM.BaumWelch();
			System.out.println("Running Evaluation Value: " + trainGLRHMM.forward());
			System.out.println();
			
		}
		System.out.println("Final Evaluation Value: " + trainGLRHMM.forward());
		int obsNumber = 1;
		while(sin.hasNext()){
			String[] line = sin.nextLine().split(",");
			EvalGLRHMM eval = new EvalGLRHMM(timeSeriesHMM, this.lamdas);
			eval.setPiMatrix(timeSeriesHMM.getPiMatrix(obsNumber));
			obsNumber += 1;
			predictFile.write(line[0]+",");
			System.out.println(line[0]);
			for(int j = 2; j < line.length; j++){
				predictFile.write(line[j]);
				if(j+1 == line.length)
					predictFile.write("\n");
				else
					predictFile.write(",");
			}
			predictFile.write(line[0]+",");
			for(int j = 2; j < line.length; j++){
				double[] p = new double[this.deltaLength+1];
				for(int s = 1; s <= this.deltaLength; s++){
					p[s] = eval.epsilon(obsLength+j, s);
				}
				int res = 1;
				for(int x = 1; x < p.length; x++){
					if(p[res] < p[x]) res = x;
				}
				predictFile.write(String.valueOf(this.lamdas[res-1]));
				if(j+1 == line.length)
					predictFile.write("\n");
				else
					predictFile.write(",");
			}
		}
		sin.close();
		predictFile.close();
	}
}
