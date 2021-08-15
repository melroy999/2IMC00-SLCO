package GeneratedCode;

import java.util.ArrayList;
import GenericCode.SharedVariable;
import GenericCode.BooleanExpression;
import GenericCode.Class;
import GenericCode.SignalMessage;
import GenericCode.ReceiveSignal;
import GenericCode.SharedVariableList;


public class ReceiveSignal_Com32Com4 extends ReceiveSignal{
	public Com sm;
	public ReceiveSignal_Com32Com4(Class slcoClass, SharedVariableList sharedVariables, Com sm) {
		super(slcoClass, sharedVariables);
		this.sm = sm;
		// TODO Auto-generated constructor stub
	}
	@Override
	public boolean evaluate() {
		// TODO Auto-generated method stub
		SignalMessage signal = slcoClass.getPortName("port_InOut").channel.peek();
		return ((String)signal.name).equals("S") && signal.args.length == 1 && true;
	}	
    @Override
	public void receive() {
		// TODO Auto-generated method stub
		SignalMessage signal = slcoClass.getPortName("port_InOut").channel.peek();
		sm.s=(String)signal.args[0];
	}	
}
