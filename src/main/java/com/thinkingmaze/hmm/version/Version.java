package com.thinkingmaze.hmm.version;

public class Version {
	private long version = 1L;
	public Version(){
		// TODO Auto-generated method stub
		this.version = 1L;
	}
	public Version(long version){
		// TODO Auto-generated method stub
		this.version = version;
	}
	public long getVersion(){
		// TODO Auto-generated method stub
		return this.version;
	}
	public void update(){
		// TODO Auto-generated method stub
		this.version += 1;
	}
	public void setVersion(long version) {
		// TODO Auto-generated method stub
		this.version = version;
	}
}
