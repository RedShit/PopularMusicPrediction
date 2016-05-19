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
	private String trainFilePath;
	private String testFilePath;
	private String predictFilePath;
	private int stuteLength;
	private double[] lamdas;
	private Random rng;
	
	public TimeSeries(String trainFilePath, String testFilePath, 
			String predictFilePath, int stuteLength) throws FileNotFoundException {
		this.trainFilePath = trainFilePath;
		this.testFilePath = testFilePath;
		this.predictFilePath = predictFilePath;
		this.stuteLength = stuteLength;
		this.rng = new Random(2234);
	}
	public void run() throws IOException {
		crossValidate(this.stuteLength);
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
	public Observation getObservation(String observation, int stateLength, boolean show) {
		this.lamdas = new double[stateLength];
		String[] obsList = observation.split(",");
		double l = Double.MAX_VALUE;
		double r = Double.MIN_VALUE;
		for(int i = 1; i < obsList.length; i++){
			l = Math.min(l, Double.parseDouble(obsList[i]));
			r = Math.max(r, Double.parseDouble(obsList[i]));
		}
		double segment = (r-l+1.0)/stateLength;
		for(int i = 0; i < this.lamdas.length; i++){
			this.lamdas[i] = l+segment*i+segment/2.0;
		}
		int[] obv = new int[obsList.length-1];
		for(int i = 0; i < obsList.length-1; i++){
			obv[i] = 1;
			for(int j = 0; j < this.lamdas.length; j++){
				obv[i] = getObvIndex(this.lamdas, Integer.parseInt(obsList[i+1]));
			}
			if(show)System.out.print(obv[i]+",");
		}
		if(show)System.out.println("");
		
		Observation obs = new Observation(obv);

		return obs;
	}
	
	private int getObvIndex(double[] lamdas, int obv){
		for(int i = 1; i < lamdas.length; i++){
			if(Math.abs(lamdas[i]-obv) > Math.abs(lamdas[i-1]-obv))
				return i;
		}
		return lamdas.length;
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

	public void crossValidate( int stateLength ) throws IOException {
		Scanner sin = new Scanner(new File(trainFilePath));
		Scanner tSin = new Scanner(new File(testFilePath));
		FileWriter predictFile = new FileWriter(new File(predictFilePath));
		while(sin.hasNext() && tSin.hasNext()){
			
			String observation = sin.nextLine();
			//System.out.println(observation);
			Observation obv = getObservation(observation, stateLength, true);
			HMM timeSeriesHMM = new GLRHMM("WeatherData", stateLength, lamdas.length);
			Training trainGLRHMM = new TrainGLRHMM(obv, timeSeriesHMM);
			System.out.println("Initial Evaluation Value: " + trainGLRHMM.forward());
			System.out.println();
			int epoch = 1;
			for (int e = 0; e < epoch; e++) {
				
				System.out.println("Training Epoch #" + (e + 1));
				trainGLRHMM.BaumWelch();
				System.out.println("Running Evaluation Value: " + trainGLRHMM.forward());
				System.out.println();
				
			}
			System.out.println("Final Evaluation Value: " + trainGLRHMM.forward());
			
			String[] line = tSin.nextLine().split(",");
			if(line[0].equals("e6e2fff03cc32ee9777de2c2ed5bac30"))
				System.out.println(line[0]);
			predictFile.write(line[0]+",");
			for(int j = 2; j < line.length; j++){
				predictFile.write(line[j]);
				if(j+1 == line.length)
					predictFile.write("\n");
				else
					predictFile.write(",");
			}
			
			for(int j = 2, pre = Integer.parseInt(line[1]); j < line.length; j++){
				EvalGLRHMM eval = new EvalGLRHMM(timeSeriesHMM);
				System.out.println(j);
				eval.trainPiMatrix(getObvIndex(this.lamdas,pre));
				double sum = 0, res = 0;
				for(int s = 0; s < this.lamdas.length; s++){
					res += this.lamdas[s]*eval.epsilon(2, s+1);
					sum += eval.epsilon(2, s+1);
				}
				res /= sum;
				pre = (int) res;
				predictFile.write(String.valueOf(pre));
				if(j+1 == line.length)
					predictFile.write("\n");
				else
					predictFile.write(",");
			}
		}
		tSin.close();
		sin.close();
		predictFile.close();
	}
}
