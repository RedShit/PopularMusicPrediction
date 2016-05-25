package com.thinkingmaze.rra.tianchi;

import java.io.FileNotFoundException;
import java.util.List;

public class Evaluate {
	public static double f1Value(List<Integer> a, List<Integer> b) throws FileNotFoundException{
		double sum = 0.0;
		double alpha = 0.0;
		int N = a.size();
		for(int i = 0; i < N; i++){
			double S = a.get(i);
			double T = b.get(i);
			if(T == 0) continue;
//			System.out.println(S + " " + T);
			alpha += (S-T)/T*(S-T)/T;
			sum += T;
		}
		alpha = 1.0-Math.sqrt(alpha/N);
		double f1 = Math.sqrt(sum)*alpha;
		System.out.println(f1 + " " + Math.sqrt(sum) + " " + (f1/Math.sqrt(sum)) + " " + N);
		return f1;
	}
}
