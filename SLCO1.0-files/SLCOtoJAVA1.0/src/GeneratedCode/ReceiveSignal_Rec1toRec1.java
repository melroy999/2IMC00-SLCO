package GeneratedCode;

import java.util.ArrayList;
import GenericCode.SharedVariable;
import GenericCode.BooleanExpression;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.ReceiveSignal;
import GenericCode.SharedVariableList;


public class ReceiveSignal_Rec1toRec1 extends ReceiveSignal{
	public Rec1 sm;
	public ReceiveSignal_Rec1toRec1(Class slcoClass, SharedVariableList sharedVariables, Rec1 sm) {
		super(slcoClass, sharedVariables);
		this.sm = sm;
		// TODO Auto-generated constructor stub
	}
	@Override
	public boolean evaluate() {
		// TODO Auto-generated method stub
		SignalMessage signal = slcoClass.getPortName("port_In1").channel.peek();
		return ((String)signal.name).equals("P") && signal.args.length == 1 && ((boolean)(signal.args[0])) == false;
	}	
    @Override
	public void receive() {
		// TODO Auto-generated method stub
		SignalMessage signal = slcoClass.getPortName("port_In1").channel.peek();
		sm.v=(boolean)signal.args[0];
	}	
}
