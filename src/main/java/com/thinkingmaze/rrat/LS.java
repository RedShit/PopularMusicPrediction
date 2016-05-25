package com.thinkingmaze.rrat;

import org.jblas.util.Random;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class LS {
	private double p;
	private double a;
	private double b;
	public LS(double p) {
		// TODO Auto-generated constructor stub
		this.p = p;
		this.a = 0;
		this.b = 0;
	}
	public double f(double x){
		return a*x + b;
	}
	public String toString(){
		String res = "LS -> ";
		res += "\t[p] = "+p+"\n";
		res += "\t\t[a] = "+a+"\n";
		res += "\t\t[b] = "+b+"\n";
		return res;
	}
}