package GeneratedCode;

import java.util.ArrayList;
import GenericCode.SharedVariable;
import GenericCode.BooleanExpression;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.SharedVariableList;


public class BooleanExpression_SendRec02SendRec1 extends BooleanExpression{
	public SendRec sm;
	public BooleanExpression_SendRec02SendRec1(Class slcoClass, SharedVariableList sharedVariables, SendRec sm) {
		super(slcoClass, sharedVariables);
		this.sm = sm;
		// TODO Auto-generated constructor stub
	}
	@Override
	public boolean evaluate() {
		// TODO Auto-generated method stub
        return ((Integer)slcoClass.getSharedVariableName("m").getValue()==6);
	}
}
