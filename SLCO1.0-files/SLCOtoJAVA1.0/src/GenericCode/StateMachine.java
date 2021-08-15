package GenericCode;

public abstract class StateMachine {
	protected Class slcoClass;

	
	public StateMachine(Class slcoClass){
		//setSlcoClass(slcoClass);	
		this.slcoClass = slcoClass;
	}

     public Class getSlcoClass() {
		return slcoClass;
	}

	public void setSlcoClass(Class slcoClass) {
		this.slcoClass = slcoClass;
	}
}
