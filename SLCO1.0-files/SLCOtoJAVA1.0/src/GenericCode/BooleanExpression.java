package GenericCode;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

public abstract class BooleanExpression extends Statement {

	public BooleanExpression(Class slcoClass, SharedVariableList sharedVariables) {
		super(slcoClass, sharedVariables);
	}

	public abstract boolean evaluate();

	// Pattern Template Method

	public boolean lockVEvaluateUnlockV(boolean isNonDeterministicTransition) {
		lockV();
		boolean isExecutedExpression = evaluate();
		if (isNonDeterministicTransition) {
			unlockV();
			return isExecutedExpression;
		}
		else {
			while (!evaluate()) {
				unlockV();
				try {
					slcoClass.lock.lock();
					slcoClass.c_lock.await(10, TimeUnit.MILLISECONDS);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} finally {
					slcoClass.lock.unlock();
				}
				lockV();
		    } 
		    unlockV();
		    return true;
		}
	}
	 
 
	/*public boolean lockVEvaluateUnlockV(boolean isNonDeterministicTransition) {
		lockV();
		boolean isExecutedExpression = evaluate();
		if (isNonDeterministicTransition) {
			unlockV();
			return isExecutedExpression;
		}
		else {
			while (!evaluate()){
				
				if (sharedVariables.size() == 1)
					try {
						sharedVariables.get(0).getC_lock_SharedVariable().await(10, TimeUnit.MILLISECONDS);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				else {
					unlockV();
					try {
						slcoClass.lock.lock();
						try {
							slcoClass.c_lock.await(10, TimeUnit.MILLISECONDS);
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					} finally {
						slcoClass.lock.unlock();
					}
					lockV();
				}
		    } 
		    unlockV();
		    return true;
		}
	}
	 */
	/*public void lockVEvaluateUnlockV() {
		lockV();
		while (!evaluate()) {
			unlockV();
			slcoClass.lock.lock();
			try {
				try {
					slcoClass.c_lock.await(2,TimeUnit.SECONDS);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			} finally {
				slcoClass.lock.unlock();
			}
			lockV();
		}
		unlockV();
	}
	*/
	/*public void lockVEvaluateUnlockV() {
		lockV();
		while (!evaluate()) {
			unlockV();
			lockV();
		}
		unlockV();
	}*/
}
