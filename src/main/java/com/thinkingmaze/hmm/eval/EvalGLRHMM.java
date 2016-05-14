package com.thinkingmaze.hmm.eval;

import java.util.Arrays;

import com.thinkingmaze.hmm.HMM;

public class EvalGLRHMM extends Evaluation {

	private double[][] tAlpha;
	private double[] PiMatrix;
	
	public double getPiMatrix(int stateIndex) {
		return PiMatrix[stateIndex-1];
	}

	public void setPiMatrix(double[] piMatrix) {
		PiMatrix = piMatrix;
	}
	
	public EvalGLRHMM(HMM hmm) {
		super.hmm = hmm;
		super.stateLength = hmm.getStateLength();
		super.deltaLength = hmm.getDeltaLength();
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
	
	public double alpha(int obsIndex, int stateIndex) {
		if(stateIndex > super.stateLength)
			throw new RuntimeException("stateIndex > super.stateLength");
		this.tAlpha = new double[obsIndex][];
		for(int i = 0; i < obsIndex; i++){
			this.tAlpha[i] = new double[super.stateLength];
			Arrays.fill(this.tAlpha[i], -1.0);
		}
		return getAlpha(obsIndex, stateIndex);
	}
	public double epsilon(int obsIndex, int deltaIndex) {
		if(deltaIndex > super.deltaLength)
			throw new RuntimeException("deltaIndex > super.deltaLength");
		double res = 0.0;
		for(int i = 1; i <= super.stateLength; i++){
			res += alpha(obsIndex,i)*hmm.getEmissionProbMatrix(i, deltaIndex);
		}
		return res;
	}

	@Override
	public double forward() {
		// TODO Auto-generated method stub
		return 0;
	}
}
