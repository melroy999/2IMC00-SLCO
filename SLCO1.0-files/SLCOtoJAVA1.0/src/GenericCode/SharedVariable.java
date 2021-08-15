package GenericCode;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

public class SharedVariable {
	private Object value;
	private Lock lock;
	private String name;
	private Condition c_lock_SharedVariable;

	public Object getValue() {
		return value;
	}

	public void setValue(Object value) {
		this.value = value;
	}

	public Lock getLock() {
		return lock;
	}

	public void setLock(Lock lock) {
		this.lock = lock;
	}
	public void lock(){
		getLock().lock();
	}
	public void unlock(){
		getLock().unlock();
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}
	
	public Condition getC_lock_SharedVariable() {
		return c_lock_SharedVariable;
	}

	public void setC_lock_SharedVariable(Condition c_lock_SharedVariable) {
		this.c_lock_SharedVariable = c_lock_SharedVariable;
	}
}
