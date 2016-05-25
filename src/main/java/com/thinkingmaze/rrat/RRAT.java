package com.thinkingmaze.rrat;

import java.util.List;

import org.jblas.util.Random;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public abstract class RRAT {
	protected LS fp;
	protected double c1;
	protected double c0;
	public RRAT(double p){
		this.fp = new LS(p);
		this.c0 = Random.nextDouble();
		this.c1 = Random.nextDouble();
	}
	public abstract double error(double x, double y);
	public abstract double error(List<Integer> data);
	public abstract double f(double x);
	public abstract String toString();
}