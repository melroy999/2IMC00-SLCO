package GenericCode;

import java.util.concurrent.TimeUnit;

//import java.util.ArrayList;

public abstract class SendSignal extends Statement {
	public SendSignal(Class slcoClass,
			SharedVariableList sharedVariables) {
		super(slcoClass, sharedVariables);
	}

	public abstract boolean send(boolean isNonDeterministicTransition);

	// Pattern Template Method
	public boolean lockVSendUnlockV(boolean isNonDeterministicTransition) {
		lockV();
		boolean isSuccessful = send(isNonDeterministicTransition);	
		unlockV();
		return isSuccessful;
	}
}
