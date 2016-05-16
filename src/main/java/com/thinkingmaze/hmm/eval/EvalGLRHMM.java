package com.thinkingmaze.hmm.eval;

import java.util.Arrays;

import com.thinkingmaze.hmm.HMM;

public class EvalGLRHMM extends Evaluation {

	private double[][] tAlpha;
	private double[] PiMatrix;
	private double[] lamdas;
	
	public double getPiMatrix(int stateIndex) {
		return PiMatrix[stateIndex-1];
	}
	
	public void setPiMatrix(int obsValue){
		double[] p = getPoissionP(obsValue);
		PiMatrix = new double[stateLength];
		double sum = 0;
		for(int i = 0; i < stateLength; i++){
			for(int j = 0; j < p.length; j++){
				if(p[j] <= 0) continue;
				PiMatrix[i] += hmm.getEmissionProbMatrix(i+1, j+1)*p[j];
			}
			sum += PiMatrix[i];
		}
		if(sum <= 0.0)
			throw new RuntimeException("setPiMatrix sum <= 0.0. ");
		for(int i = 0; i < stateLength; i++){
			PiMatrix[i] /= sum;
		}
	}
	
	public void setPiMatrix(double[] piMatrix) {
		PiMatrix = piMatrix;
	}
	
	public EvalGLRHMM(HMM hmm, double[] lamdas) {
		super.hmm = hmm;
		super.stateLength = hmm.getStateLength();
		super.deltaLength = hmm.getDeltaLength();
		this.lamdas = lamdas;
	}
	
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
	
	private double[] getPoissionP(int d) {
		// TODO Auto-generated method stub
		double[] p = new double[this.lamdas.length];
		double sum = 0;
		for(int i = 0; i < this.lamdas.length; i++){
			p[i] = getPoisson(this.lamdas[i],d);
			sum += p[i];
		}
		if(sum <= 0.0)
			throw new NullPointerException("getDelta: sum <= 0.0.");
		for(int i = 0; i < this.lamdas.length; i++){
			p[i] = p[i]/sum;
		}
		return p;
	}
	
	private double getAlpha(int obsIndex, int stateIndex){
		if(this.tAlpha[obsIndex-1][stateIndex-1]>=0)
			return this.tAlpha[obsIndex-1][stateIndex-1];
		if (obsIndex == 1) {
			return this.tAlpha[obsIndex-1][stateIndex-1]=getPiMatrix(stateIndex);
		}
		
		double sum = 0;

		for (int j = 1; j <= stateLength; j++) {
			sum += hmm.getStateTransitProbMatrix(j, stateIndex)
					* getAlpha(obsIndex - 1 , j);
		}
		return this.tAlpha[obsIndex-1][stateIndex-1]=sum;
	}
	
	private double alpha(int obsIndex, int stateIndex) {
		if(stateIndex > super.stateLength)
			throw new RuntimeException("stateIndex > super.stateLength");
		
		return getAlpha(obsIndex, stateIndex);
	}
	public double epsilon(int obsIndex, int deltaIndex) {
		if(deltaIndex > super.deltaLength)
			throw new RuntimeException("deltaIndex > super.deltaLength");
		this.tAlpha = new double[obsIndex][];
		for(int i = 0; i < obsIndex; i++){
			this.tAlpha[i] = new double[super.stateLength];
			Arrays.fill(this.tAlpha[i], -1.0);
		}
		double res = 0.0;
		for(int i = 1; i <= super.stateLength; i++){
			res += alpha(obsIndex,i)*hmm.getEmissionProbMatrix(i, deltaIndex);
		}
		return res;
	}
}
