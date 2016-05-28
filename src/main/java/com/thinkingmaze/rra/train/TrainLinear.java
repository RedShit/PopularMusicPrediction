package com.thinkingmaze.rra.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.thinkingmaze.rrat.LS;

public class TrainLinear extends Training {
	
	public TrainLinear(LS ls) {
		super.ls = ls;
	}
	
	@Override
	public void trainA(List<Integer> data) {
		// TODO Auto-generated method stub
		double avrY = 0, avrX = 0;
		for(int x = 0; x < data.size(); x++){
			avrY += data.get(x);
			avrX += x;
		}
		avrY /= data.size();
		avrX /= data.size();
		
		double aZ = 0, aM = 0;
		for(int x = 0; x < data.size(); x++){
			double X = x - avrX;
			double Y = data.get(x)-avrY;
			aZ += X*Y;
			aM += X*X;
		}
		ls.setA(aZ/aM);
	}

	@Override
	public void trainB(List<Integer> data) {
		// TODO Auto-generated method stub
		List<Double> res = new ArrayList<Double>();
		for(int x = 0; x < data.size(); x++){
			res.add(data.get(x) - ls.getA()*x);
		}
		Collections.sort(res);
		ls.setB(res.get(res.size()/2));
	}
	
	@Override
	public double output(int x) {
		// TODO Auto-generated method stub
		return ls.f(x);
	}
	
	private static double variance(List<Double> data){
		List<Double> tData = new ArrayList<Double>();
		for(int i = 0; i < data.size(); i++){
			tData.add(data.get(i));
		}
		double maxValue = Integer.MIN_VALUE;
		double avr = 0;
		Collections.sort(tData);
		maxValue = tData.get(2)+1;
		for(int x = 0; x < tData.size(); x++){
			double value = tData.get(x)/maxValue;
			avr += value;
		}
		avr/=tData.size();
		double var = 0;
		for(int x = 0; x < tData.size(); x++){
			double value = tData.get(x)/maxValue;
			var += (value-avr)*(value-avr);
		}
		var = Math.sqrt(var);
		var /= data.size();
		return var;
	}

	@Override
	public double error(List<Integer> data) {
		// TODO Auto-generated method stub
		List<Double> tData = new ArrayList<Double>();
		for(int x = 0; x < data.size(); x++){
			tData.add(data.get(x)-output(x));
		}
		return variance(tData);
	}

	public double error(List<Integer> data, int x) {
		// TODO Auto-generated method stub
		double res = 0;
		for(int i = 0; i < data.size(); i++){
			res += 1.0*Math.abs(x-data.get(i))/data.get(i);
		}
		return 1.0-res/data.size();
	}
	
	@Override
	public double median(List<Integer> data) {
		// TODO Auto-generated method stub
		List<Integer> noise = new ArrayList<Integer>();
		int lValue = Integer.MAX_VALUE;
		int rValue = Integer.MIN_VALUE;
		for(int x = 0; x < data.size(); x++){
			noise.add(data.get(x));
			lValue = (int) Math.min(lValue, data.get(x));
			rValue = (int) Math.max(rValue, data.get(x));
		}
		int res = 0;
		Collections.sort(noise);
		for(int x = lValue; x <= rValue; x++){
			if(error(noise, x) > res)
				res = x;
		}
		return res;
	}
	
}
