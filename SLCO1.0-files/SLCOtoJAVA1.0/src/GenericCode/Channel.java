package GenericCode;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public abstract class Channel {
	protected ArrayBlockingQueue<SignalMessage> asynQueue;
	//protected Object senderLock;
	//protected Object receiverLock;
	protected Lock senderLock;
	protected Condition c_senderLock;
	
	protected Lock receiverLock;
    protected Condition c_receiverLock;
    
	public Channel() {
		
		//senderLock = new Object();
		//receiverLock = new Object();
		//asynQueue = new ArrayBlockingQueue<SignalMessage>(1);
		senderLock = new ReentrantLock();
		c_senderLock =senderLock.newCondition(); 
		
		receiverLock = new ReentrantLock();
		c_receiverLock = receiverLock.newCondition();
	}
	public abstract boolean send(SendSignal sendSignalName, String t,
			boolean isNonDeterministicTransition);

	public abstract boolean put(SignalMessage signal, boolean isNonDeterministicTransition);

	public abstract SignalMessage receive(ReceiveSignal expression,
			String transitionName, boolean isNonDeterministicTransition);

	public SignalMessage peek() {
		SignalMessage signal = asynQueue.peek();
		return signal;
	}
}
