package com.thinkingmaze.rra.train;

import java.util.List;

import com.thinkingmaze.rrat.LS;

public abstract class Training {
	protected LS ls;
	public abstract void train(List<Integer> data);
	public abstract double output(int x);
	public abstract double error(List<Integer> data);
	public abstract double median(List<Integer> data);
}