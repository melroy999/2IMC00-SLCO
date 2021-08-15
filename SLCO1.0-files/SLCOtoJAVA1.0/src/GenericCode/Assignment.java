package GenericCode;

import java.util.concurrent.TimeUnit;

//import java.util.ArrayList;

public abstract class Assignment extends Statement {
	public Assignment(Class slcoClass,
			SharedVariableList sharedVariables) {
		super(slcoClass, sharedVariables);
	}

	public abstract void assign();

	// Pattern Template Method
	public void lockVAssignUnlockV() {
		lockV();
		assign();	
		//sharedVariables.get(index).getC_lock_SharedVariable().signalAll();
		unlockV();
		try {
			slcoClass.lock.lock();
			slcoClass.c_lock.signalAll();
		} finally {
			slcoClass.lock.unlock();
		 }
		//unlockV();
	}
	/*public void lockVAssignUnlockV() {
		lockV();
		assign();
		unlockV();
	}*/
}
