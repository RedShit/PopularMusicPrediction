package com.thinkingmaze.hmm.observation;

import java.util.Iterator;


public class NullIterator implements Iterator<Object> {

	public Object next() {
		return null;
	}
	
	public boolean hasNext() {
		return false;
	}
	
	public void remove() {
		throw new UnsupportedOperationException();
	}
}
