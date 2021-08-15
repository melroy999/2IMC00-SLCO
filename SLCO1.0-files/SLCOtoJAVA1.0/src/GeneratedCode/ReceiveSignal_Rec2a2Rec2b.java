package GeneratedCode;

import java.util.ArrayList;
import GenericCode.SharedVariable;
import GenericCode.BooleanExpression;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.ReceiveSignal;
import GenericCode.SharedVariableList;


public class ReceiveSignal_Rec2a2Rec2b extends ReceiveSignal{
	public Rec2 sm;
	public ReceiveSignal_Rec2a2Rec2b(Class slcoClass, SharedVariableList sharedVariables, Rec2 sm) {
		super(slcoClass, sharedVariables);
		this.sm = sm;
		// TODO Auto-generated constructor stub
	}
	@Override
	public boolean evaluate() {
		// TODO Auto-generated method stub
		SignalMessage signal = slcoClass.getPortName("port_In2").channel.peek();
		return ((String)signal.name).equals("Q") && signal.args.length == 1 && ((Integer)(signal.args[0])) >= 0;
	}	
    @Override
	public void receive() {
		// TODO Auto-generated method stub
		SignalMessage signal = slcoClass.getPortName("port_In2").channel.peek();
		slcoClass.getSharedVariableName("m").setValue((Integer)signal.args[0]);
	}	
}
