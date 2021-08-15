package GenericCode;

public abstract class Statement {
	protected Class slcoClass;
	protected SharedVariableList sharedVariables;

	public Statement(Class slcoClass,
			SharedVariableList sharedVariables) {
		this.slcoClass = slcoClass;
		this.sharedVariables = sharedVariables;
	}

	public void lockV() {
		for (int j = 0; j <= sharedVariables.size() - 1; j++) {
			sharedVariables.get(j).lock();
		}
	}

	public void unlockV() {
		for (int j = 0; j <= sharedVariables.size() - 1; j++) {
			sharedVariables.get(j).unlock();
		}
	}
}

