package com.thinkingmaze.hmm.observation;

import java.util.Iterator;
import java.util.Stack;

public class ObsIterator implements Iterator<Object> {
	@SuppressWarnings("rawtypes")
	Stack<Iterator> stack = new Stack<Iterator>();
	
	public ObsIterator(Iterator<ObserveComponent> iterator) {
		stack.push(iterator);
	}
	
	public Object next() {
		if (hasNext()) {
			Iterator<?> iterator = stack.peek();
			ObserveComponent component = (ObserveComponent)iterator.next();
			
			if (component instanceof Observations) {
				stack.push(component.createIterator());
			}
			
			return component;
		} else {
			return null;
		}
	}
	
	public boolean hasNext() {
		if (stack.empty()) {
			return false;
		} else {
			Iterator<?> iterator = stack.peek();
			if (!iterator.hasNext()) {
				stack.pop();
				return hasNext();
			} else {
				return true;
			}
		}
	}
	
	public void remove() {
		throw new UnsupportedOperationException();
	}
}
