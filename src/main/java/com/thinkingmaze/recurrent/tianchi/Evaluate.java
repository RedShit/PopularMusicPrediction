package com.thinkingmaze.recurrent.tianchi;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Evaluate {
	private static final String predictFilePath = "D:/MyEclipse/tianchi/target/mars_tianchi_predict_data.csv";
	public static void f1Value(String predictFilePath) throws FileNotFoundException{
		int artistNum = 50;
		int predictDays = 60;
		Scanner sin = new Scanner(new File(predictFilePath));
		double[][] predict = new double[artistNum*2][];
		int currIdx = 0;
		while(sin.hasNextLine()){
			predict[currIdx] = new double[predictDays];
			String[] strs = sin.nextLine().split(",");
			for(int i = 0; i < predictDays; i++)
				predict[currIdx][i] = Double.valueOf(strs[i]);
			currIdx +=1;
		}
		double f1 = 0.0;
		double optF1 = 0.0;
		for(int i = 0; i < artistNum; i++){
			int index = i*2;
			double alpha = 0.0;
			int N = predictDays;
			double sum = 0;
			for(int j = 0; j < N; j++){
				double S = predict[index+1][j];
				double T = predict[index][j];
				if(T == 0) continue;
				alpha = alpha + ((S-T)/T)*((S-T)/T);
				sum = sum + T;
			}
			alpha = Math.sqrt(alpha/N);
			f1 = f1 + (1.0-alpha)*Math.sqrt(sum);
			optF1 = optF1 + Math.sqrt(sum);
		}
		sin.close();
		System.out.println(f1 + " " + optF1 + " " + (100.0*f1/optF1) + "%");
	}
	public static void main(String[] args) throws FileNotFoundException{
		f1Value(predictFilePath);
	}
}
