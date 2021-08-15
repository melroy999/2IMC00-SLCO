package GenericCode;

import java.util.ArrayList;

public abstract class ReceiveSignal extends Statement {
	// public SLCOClass slcoClass;
	public ReceiveSignal(Class slcoClass, SharedVariableList sharedVariables) {
		// this.slcoClass = slcoClass;
		super(slcoClass, sharedVariables);
	}

	public abstract void receive();

	public abstract boolean evaluate();

	// Pattern Template Method for receiving signalMessage
	public boolean lockVEvaluateReceiveUnlockV() {
		lockV();
		boolean isExecutedStatement = evaluate();
		
		if (isExecutedStatement) {
			receive();
			slcoClass.lock.lock();
			try {
				slcoClass.c_lock.signalAll();
			} finally {
				slcoClass.lock.unlock();
			}
		}
		unlockV();
		return isExecutedStatement;
	}
}
