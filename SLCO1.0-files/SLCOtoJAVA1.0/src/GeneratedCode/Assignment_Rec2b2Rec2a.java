package GeneratedCode;

import java.util.ArrayList;
import GenericCode.SharedVariable;
import GenericCode.SharedVariableList;

import GenericCode.Assignment;
import GenericCode.Class;
public class Assignment_Rec2b2Rec2a extends Assignment{
	public Rec2 sm;
	public Assignment_Rec2b2Rec2a(Class slcoClass, SharedVariableList sharedVariables, Rec2 sm) {
		super(slcoClass, sharedVariables);
		this.sm = sm;
		// TODO Auto-generated constructor stub
	}
	@Override
	public void assign() {
		// TODO Auto-generated method stub
			slcoClass.getSharedVariableName("m").setValue(((Integer)slcoClass.getSharedVariableName("m").getValue()+1));
	}
}
