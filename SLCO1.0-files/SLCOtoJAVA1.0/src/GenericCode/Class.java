package GenericCode;

import java.util.ArrayList;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

//public class SLCOClass {

public class Class {
	private String className;
	private String objectName;
	private Model classParent;
	
	protected Lock lock;
	protected Condition c_lock;

	//protected ArrayList<SharedVariable> sharedVariables = new ArrayList<SharedVariable>();
	//protected SharedVariable[] sharedVariables;
	
	protected SharedVariableList sharedVariables;// = new SharedVariableList();
	protected ArrayList<Port> ports;// = new ArrayList<Port>();
	protected ArrayList<StateMachine> stateMachines;// = new ArrayList<StateMachine>();
	
	public Class(String className, String objectName, Model parent) {
		
		setClassName(className);
		setObjectName(objectName);
		setClassParent(parent);
		
		sharedVariables = new SharedVariableList();
		ports = new ArrayList<Port>();
		stateMachines = new ArrayList<StateMachine>();
		
		lock = new ReentrantLock();
		c_lock = lock.newCondition();
	}

	public String getClassName() {
		return className;
	}

	public void setClassName(String className) {
		this.className = className;
	}

	public String getObjectName() {
		return objectName;
	}

	public void setObjectName(String objectName) {
		this.objectName = objectName;
	}

	public Model getClassParent() {
		return classParent;
	}

	public void setClassParent(Model classParent) {
		this.classParent = classParent;
	}

	public ArrayList<Port> getPorts() {
		return ports;
	}

	public void setPorts(ArrayList<Port> ports) {
		this.ports = ports;
	}

	public ArrayList<StateMachine> getStateMachines() {
		return stateMachines;
	}

	public void setStateMachines(ArrayList<StateMachine> stateMachines) {
		this.stateMachines = stateMachines;
	}

	public SharedVariableList getSharedVariables() {
		return sharedVariables;
	}

	public void setSharedVariables(SharedVariableList sharedVariables) {
		this.sharedVariables = sharedVariables;
	}

	public Port getPortName(String name) {
		for (Port p_Name : ports) {
			if (name.compareTo(p_Name.getName()) == 0) {
				return p_Name;
			}
		}
		return null;
	}

	public SharedVariable getSharedVariableName(String name) {
		int length = sharedVariables.size();
		for(int i = 0; i < length; i++){
			if (name.compareTo(sharedVariables.get(i).getName()) == 0) {
				return sharedVariables.get(i);
			}
		}
		/*for (SharedVariable p_Name : sharedVariables) {
			if (name.compareTo(p_Name.getName()) == 0) {
				return p_Name;
			}
		}*/
		return null;
	}
}
