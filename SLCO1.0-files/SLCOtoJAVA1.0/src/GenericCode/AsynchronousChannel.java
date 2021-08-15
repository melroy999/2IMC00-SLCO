package GenericCode;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

public class AsynchronousChannel extends Channel {

	public AsynchronousChannel() {
		asynQueue = new ArrayBlockingQueue<SignalMessage>(1);
	}

	@Override
	public boolean send(SendSignal sendSignal,
			String transitionName, boolean isNonDeterministicTransition) {
		
		// TODO Auto-generated method stub
		///////////////////////////////////////////////////
		try{
			senderLock.lock();
			SignalMessage signal = peek();
			if (isNonDeterministicTransition) {
				if (signal == null) {
					sendSignal.lockVSendUnlockV(true);
					System.out.println("Transition: " + transitionName);
					
					try{ 
						receiverLock.lock();
						c_receiverLock.signalAll();
					}finally{
						receiverLock.unlock();
					}
					return true;
				} else {
					return false;
				}
			} else {
				sendSignal.lockVSendUnlockV(false);
				System.out.println("Transition: " + transitionName);
				
				try{ 
					receiverLock.lock();
					c_receiverLock.signalAll();
				}finally{
					receiverLock.unlock();
				}
				
				return true;
			}

		} finally{
			senderLock.unlock();
		}
	}

	@Override
	public boolean put(SignalMessage signal, boolean isNonDeterministicTransition) {
		// TODO Auto-generated method stub
		try {
			asynQueue.put(signal);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return true;
	}

	@Override
	public SignalMessage receive(ReceiveSignal receiveSignal,
			String transitionName, boolean isNonDeterministicTransition){
		// TODO Auto-generated method stub
		boolean isExecutableStatement = false;
		SignalMessage signal = null;
		try {
			receiverLock.lock();
			
		while (!isExecutableStatement) {	
				signal = peek();
				if (signal != null) {
					isExecutableStatement = receiveSignal
								.lockVEvaluateReceiveUnlockV();
					if (isExecutableStatement) {
						signal = (SignalMessage) asynQueue.take();
						System.out.println("Transition: " + transitionName);
						return signal;
					} else {
						if (isNonDeterministicTransition) {
							signal = null;
							return signal;
						} else {
							c_receiverLock.await(10, TimeUnit.MILLISECONDS);
						}
					}
				} else {
					if (isNonDeterministicTransition) {
						signal = null;
						return signal;
					}
					else {
						c_receiverLock.await(10, TimeUnit.MILLISECONDS);
					}
				}
		}
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			receiverLock.unlock();
		}
		return signal;
	}
}
