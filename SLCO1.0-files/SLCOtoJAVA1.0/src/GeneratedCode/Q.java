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


public class Q extends Class{
    
	public Port port_Out1;
	public Port port_Out2;
	public Port port_InOut;
	
	public Com sm_Com;
	
    public Q(String name, String objectName, Model slcoModel, Channel channel_Out1, Channel channel_Out2, Channel channel_InOut) {
	    super(name, objectName, slcoModel);
    
        port_Out1 = new Port(channel_Out1);
        port_Out1.setName("port_Out1");
        ports.add(port_Out1);       
        port_Out2 = new Port(channel_Out2);
        port_Out2.setName("port_Out2");
        ports.add(port_Out2);       
        port_InOut = new Port(channel_InOut);
        port_InOut.setName("port_InOut");
        ports.add(port_InOut);       
    
        Com sm_Com = new Com(this);
		new Thread(sm_Com).start();
    
    }
}