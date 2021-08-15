package GeneratedCode;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import GenericCode.Channel;
import GenericCode.Port;
import GenericCode.Class;
import GenericCode.Model;
import GenericCode.SharedVariable;
import GenericCode.SharedVariableList;


public class P extends Class{
	public SharedVariable m;
    
	public Port port_In1;
	public Port port_In2;
	public Port port_InOut;
	
	public Rec1 sm_Rec1;
	public Rec2 sm_Rec2;
	public SendRec sm_SendRec;
	
    public P(String name, String objectName, Model slcoModel, Channel channel_In1, Channel channel_In2, Channel channel_InOut) {
	    super(name, objectName, slcoModel);
        m = new SharedVariable();
        m.setName("m");
        m.setValue(0);
        Lock lock_m = new ReentrantLock();
        m.setLock(lock_m); 
        Condition c_lock_m = lock_m.newCondition();
    	m.setC_lock_SharedVariable(c_lock_m);       
        sharedVariables.add(m);
    
        port_In1 = new Port(channel_In1);
        port_In1.setName("port_In1");
        ports.add(port_In1);       
        port_In2 = new Port(channel_In2);
        port_In2.setName("port_In2");
        ports.add(port_In2);       
        port_InOut = new Port(channel_InOut);
        port_InOut.setName("port_InOut");
        ports.add(port_InOut);       
    
        Rec1 sm_Rec1 = new Rec1(this);
        Rec2 sm_Rec2 = new Rec2(this);
        SendRec sm_SendRec = new SendRec(this);
		new Thread(sm_Rec1).start();
		new Thread(sm_Rec2).start();
		new Thread(sm_SendRec).start();
    
    }
}