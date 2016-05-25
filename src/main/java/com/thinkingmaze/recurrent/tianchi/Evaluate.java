package com.thinkingmaze.recurrent.tianchi;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Evaluate {
	public static double f1Value(String predictFilePath) throws FileNotFoundException{
		int artistNum = 1;
		int predictDays = 60;
		Scanner sin = new Scanner(new File(predictFilePath));
		double[][] predict = new double[artistNum*3][];
		int currIdx = 0;
		while(sin.hasNextLine()){
			predict[currIdx] = new double[predictDays];
			String line = sin.nextLine();
//			System.out.println(line);
			String[] strs = line.split(",");
			for(int i = 0; i < strs.length; i++)
				predict[currIdx][i] = Double.valueOf(strs[i]);
			currIdx +=1;
		}
		double f1 = 0.0;
		double optF1 = 0.0;
		double midF1 = 0.0;
		for(int i = 0; i < artistNum; i++){
//			if(i == 26||i==39) continue;
			int index = i*3;
			double alpha = 0.0;
			int N = predictDays;
			double sum = 0;
			for(int j = 0; j < N; j++){
				double S = Math.min(predict[index+2][j], predict[index+1][j]*1.1);
				double T = predict[index][j];
				if(T == 0) continue;
				
				alpha = alpha + ((S-T)/T)*((S-T)/T);
				sum = sum + T;
			}
			alpha = Math.sqrt(alpha/N);
//			System.out.println(i + " " + (1-alpha) + " " + Math.sqrt(sum));
			f1 = f1 + (1.0-alpha)*Math.sqrt(sum);
			optF1 = optF1 + Math.sqrt(sum);
			sum = 0; alpha = 0;
			for(int j = 0; j < N; j++){
				double S = predict[index+1][j];
				double T = predict[index][j];
				if(T == 0) continue;
				
				alpha = alpha + ((S-T)/T)*((S-T)/T);
				sum = sum + T;
			}
			alpha = Math.sqrt(alpha/N);
			midF1 = midF1 + (1.0-alpha)*Math.sqrt(sum);
		}
		sin.close();
		return f1;
//		return (f1 + " " + midF1 + " " + optF1);
	}
	public static void main(String[] args) throws IOException{  
		Scanner artistSin = new Scanner(new File("D:/MyEclipse/alibaba/mars_tianchi_artist_id.csv"));
		String resultFilePath = "D:/MyEclipse/alibaba/result.csv";
		FileWriter resultFile = new FileWriter(new File(resultFilePath));
		double f1 = 0;
		while(artistSin.hasNext()){
			String artistId = artistSin.next();
//			resultFile.write(artistId+",");
			String predictFilePath = "D:/MyEclipse/alibaba/"+artistId+".csv";
			f1 += f1Value(predictFilePath);
//			Scanner sin = new Scanner(new File(predictFilePath));
//			resultFile.write(sin.next()+"\n");
//			sin.close();
		}
		System.out.println(f1/8003);
		resultFile.close();
		artistSin.close();
	}
}
