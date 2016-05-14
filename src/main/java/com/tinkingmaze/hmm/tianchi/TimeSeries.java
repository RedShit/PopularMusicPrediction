package com.tinkingmaze.hmm.tianchi;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import com.thinkingmaze.hmm.GLRHMM;
import com.thinkingmaze.hmm.HMM;
import com.thinkingmaze.hmm.observation.Observation;
import com.thinkingmaze.hmm.observation.Observations;
import com.thinkingmaze.hmm.train.TrainGLRHMM;
import com.thinkingmaze.hmm.train.Training;

public class TimeSeries {
	private String trainFilePath;
	private String testFilePath;
	private int obsNumber;
	private int obsLength;
	private final int deltaLength;
	private double[][] real;
	private double[] lamda;
	private int l;
	private int r;
	
	public TimeSeries(String trainFilePath, String testFilePath, int obsNumber, int obsLength, int deltaLength) {
		this.trainFilePath = trainFilePath;
		this.testFilePath = testFilePath;
		this.obsNumber = obsNumber;
		this.obsLength = obsLength;
		this.deltaLength = deltaLength;
		this.lamda = new double[deltaLength];
		this.l = 0;
		this.r = 500;
		double segment = (r-l+1.0)/(deltaLength-1);
		for(int i = 0; i < deltaLength; i++){
			this.lamda[i] = l+i*segment+segment/2;
		}
	}
	
	public void run(String predictFilePath) throws IOException {
		Observations obs = getObservation(trainFilePath, obsNumber, obsLength, false);
		crossValidate(predictFilePath, testFilePath, obs, 10, this.deltaLength);
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
		this.real = new double[obsNumber][obsLength];
		try {
			File file = new File(filename);
			Scanner scan = new Scanner(file);
			if(filename.endsWith(".csv")){
				for(int i = 0; i < result.length; i++){
					String[] lines = scan.nextLine().split(",");
					double segment = (r-l+1.0)/(deltaLength-1);
					for (int j = 0; j < result[0].length; j++){
						real[i][j] = Double.parseDouble(lines[j+1]);
						int temp = (int) ((Double.parseDouble(lines[j+1])-l)/segment + 1);
						if(temp > 50)
						if(temp > deltaLength)
							temp = deltaLength;
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

	@SuppressWarnings("unused")
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

	public void crossValidate(String predictFilePath, String testFilePath, Observations obs, 
			int stateLength, int delta) throws IOException {
		Scanner sin = new Scanner(new File(testFilePath));
		FileWriter predictFile = new FileWriter(new File(predictFilePath));
		HMM timeSeriesHMM = new GLRHMM("WeatherData", stateLength, deltaLength, obs.getSize());
		Training trainGLRHMM = new TrainGLRHMM(obs, timeSeriesHMM);
		System.out.println("Initial Evaluation Value: " + trainGLRHMM.forward());
		System.out.println();
		int epoch = 50;
		for (int e = 0; e < epoch; e++) {

			System.out.println("Training Observations #" + (e + 1));
			trainGLRHMM.BaumWelch();
			System.out.println("Running Evaluation Value: " + trainGLRHMM.forward());
			System.out.println();

			String[] line = sin.nextLine().split(",");
			int[] testData = new int[line.length];
			for(int j = 1; j < testData.length; j++){
				testData[j] = Integer.parseInt(line[j]);
				predictFile.write(line[j]);
				if(j+1 == testData.length)
					predictFile.write("\n");
				else
					predictFile.write(",");
			}
			for(int j = 1; j < testData.length; j++){
				testData[j] = Integer.parseInt(line[j]);
				predictFile.write(line[j]);
				if(j+1 == testData.length)
					predictFile.write("\n");
				else
					predictFile.write(",");
			}
		}
		System.out.println("Final Evaluation Value: " + trainGLRHMM.forward());
		sin.close();
		predictFile.close();
	}
}
