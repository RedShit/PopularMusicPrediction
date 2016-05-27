package com.thinkingmaze.rra.tianchi;


import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import com.thinkingmaze.rra.train.TrainLinear;
import com.thinkingmaze.rra.train.Training;
import com.thinkingmaze.rrat.LS;

public class RRAModelling {
	//Median number
    public static final int median = 2;
	
	public static void main( String[] args ) throws Exception {
		Scanner artistSin = new Scanner(new File("D:/MyEclipse/alibaba/mars_tianchi_artist_id.csv"));
		String filePath = "D:/MyEclipse/alibaba/mars_tianchi_data.csv";
		ArtistIterator iter = new ArtistIterator(filePath, "all");
		double f1 = 0, f2 = 0, f3 = 0.0;
		while(artistSin.hasNext()){
			String artistId = "5e2ef5473cbbdb335f6d51dc57845437"; artistSin.next();
			String predictFilePath = "D:/MyEclipse/alibaba/"+artistId+".csv";
			FileWriter predictFile = new FileWriter(new File(predictFilePath));
			LS ls = new LS(0.5);
			Training model = new TrainLinear(ls);
			List<Integer> data = iter.getActionList(artistId);
			List<Integer> trainAData = new ArrayList<Integer>();
			List<Integer> trainBData = new ArrayList<Integer>();
			List<Integer> testData = new ArrayList<Integer>();
			List<Integer> mid = new ArrayList<Integer>();
			List<Integer> predictData = new ArrayList<Integer>();
			int N = data.size()-1;
			for(int i = 0; i < data.size(); i++){
				if(i>=N-70 && i<=N-60) {
					mid.add(data.get(i));
				}
				if(i>=N-70 && i<=N-60) {
					trainBData.add(data.get(i));
				}
				
				if(i<N-90) continue;
				
				if(i<=N-60) trainAData.add(data.get(i));
				else testData.add(data.get(i));
			}
			
			trainAData = curve(trainAData);
			trainBData = curve(trainBData);
			for(int i = 0; i < trainAData.size(); i++){
				System.out.print(data.get(i));
				System.out.print(i+1==trainAData.size()?"\n":",");
			}
			Collections.sort(trainAData);
			Collections.sort(mid);
			int t = 0;
			if(variance(trainAData) > 0.2) t=2;
			System.out.println(artistId);
			System.out.println("Variance is " + variance(trainAData));
			double e1 = model.error(trainBData);
			System.out.println("Before Train Error is " + e1);
			model.trainA(trainAData);
			model.trainB(trainBData);
			double e2 = model.error(trainBData);
			System.out.println("After Train Error is " + e2);
			System.out.print(mid.get(5)+" "+mid.get(4)+" "+model.output(5)+"\n");
			for(int i = 0; i < data.size(); i++){
				if(i<=N-60) continue;
				if(e1>e2) predictData.add((int) (model.output(2-t)+0.5));
				else predictData.add(mid.get(4));
			}
			f1 += Evaluate.f1Value(predictData, testData);
			predictData = new ArrayList<Integer>();
			for(int i = 0; i < data.size(); i++){
				if(i<=N-60) continue;
				predictFile.write(String.valueOf(mid.get(5-t)));
				predictFile.write(i+1==data.size()?"\n":",");
				predictData.add(mid.get(4-t));
			}
			predictFile.close();
			f2 += Evaluate.f1Value(predictData, testData);
			predictData = new ArrayList<Integer>();
			for(int i = 0; i < data.size(); i++){
				if(i<=N-60) continue;
				predictData.add(mid.get(5));
			}
			f3 += Evaluate.f1Value(predictData, testData);
			System.out.println(ls.toString());
			System.out.println("");
			break;
		}
		System.out.println("F1 is " + f1 + ", F2 is " + f2 + ", f3 is " + f3 + ", " + (f2/f1*100.0));
		artistSin.close();
		System.out.println("\n\n complete");
	}
	private static int curve(List<Integer> data, int x){
    	List<Integer> t = new ArrayList<Integer>();
    	for(int i = Math.max(0, x-median); i < Math.min(x+median+1, data.size()); i++){
    		t.add(data.get(i));
    	}
    	Collections.sort(t);
    	return t.get((t.size()-1)/2);
    }
	private static List<Integer> curve(List<Integer> data){
		List<Integer> res = new ArrayList<Integer>();
    	for(int i = median; i < data.size()-median; i++){
    		res.add(curve(data, i));
    	}
    	return res;
    }
	private static double variance(List<Integer> data){
		double maxValue = Integer.MIN_VALUE;
		double avr = 0;
		Collections.sort(data);
		maxValue = data.get(2)+1;
		for(int x = 0; x < data.size(); x++){
			double value = data.get(x)/maxValue;
			avr += value;
		}
		avr/=data.size();
		double var = 0;
		for(int x = 0; x < data.size(); x++){
			double value = data.get(x)/maxValue;
			var += (value-avr)*(value-avr);
		}
		var = Math.sqrt(var);
		var /= data.size();
		return var;
	}
}