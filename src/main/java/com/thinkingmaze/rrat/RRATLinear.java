package com.thinkingmaze.rrat;

import java.util.List;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class RRATLinear extends RRAT{

	public RRATLinear(double p) {
		super(p);
		// TODO Auto-generated constructor stub
	}
	@Override
	public double error(double x, double y) {
		// TODO Auto-generated method stub
		double p = super.fp.getP();
		double f = super.fp.getA()*x + super.fp.getB();
		if(y >= f) return p*(y-f);
		return (1-p)*(f-y);
	}
	public double f(double x){
		return super.fp.f(x)*super.c1 + super.c0;
	}
	@Override
	public double error(List<Integer> data) {
		// TODO Auto-generated method stub
		double res = 0.0;
		for(int x = 0; x < data.size(); x++){
			res += error(x, data.get(x));
		}
		return res;
	}
	@Override
	public String toString() {
		// TODO Auto-generated method stub
		String res = "[RRAT]:\n";
		res += "[fp] = "+super.fp.toString()+"\n";
		res += "[c1] = "+super.c1+"\n";
		res += "[c0] = "+super.c0+"\n";
		return res;
	}
}