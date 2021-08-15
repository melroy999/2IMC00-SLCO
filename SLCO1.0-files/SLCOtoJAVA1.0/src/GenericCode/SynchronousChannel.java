package GenericCode;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

public class SynchronousChannel extends Channel {
	protected SynchronousQueue<SignalMessage> synQueue;

	public SynchronousChannel() {
		synQueue = new SynchronousQueue<SignalMessage>();
		asynQueue = new ArrayBlockingQueue<SignalMessage>(1);
	}

	@Override
	public boolean send(SendSignal sendSignal, String transitionName, 
			boolean isNonDeterministicTransition) {
		// TODO Auto-generated method stub
		try{
			senderLock.lock();
			if (isNonDeterministicTransition) {
				boolean isExecutedStatement = sendSignal.lockVSendUnlockV(true);
				if (isExecutedStatement) {
					System.out.println("Transition: " + transitionName);
					return true;
				} else {
					return false;	
				}
			} else {
				sendSignal.lockVSendUnlockV(false);
				System.out.println("Transition: " + transitionName);
                return true;
			}
		} finally{
			senderLock.unlock();
		}
	}
	@Override
	public boolean put(SignalMessage signal, boolean isNonDeterministicTransition) {
		// TODO Auto-generated method stub
		boolean isTaken = false;
		try {
			asynQueue.put(signal);
			receiverLock.lock();
			c_receiverLock.signalAll();
			receiverLock.unlock();
			
			if (isNonDeterministicTransition){
				
				isTaken = synQueue.offer(signal, 10, TimeUnit.MILLISECONDS);
				
				receiverLock.lock();
				asynQueue.take();
				receiverLock.unlock();	
				
				if (isTaken) {
					return true;
				} else {
					return false;
				}
				
			} else {
				
				synQueue.put(signal);
				
				receiverLock.lock();
				asynQueue.take();
				receiverLock.unlock();	
				return true;
			}

		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return isTaken;
	}
	@Override
	public SignalMessage receive(ReceiveSignal receiveSignal, String transitionName,
			boolean isNonDeterministicTransition) {
		// TODO Auto-generated method stub
		boolean isExecutableStatement = false;
		SignalMessage signal = null;
		while (!isExecutableStatement) {	
			try {
				receiverLock.lock();
				signal = peek();
				if (signal != null) {
						isExecutableStatement = receiveSignal
								.lockVEvaluateReceiveUnlockV();
						if (isExecutableStatement) {
							signal = (SignalMessage) synQueue.poll(10, TimeUnit.MILLISECONDS);
							if (signal != null) {
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
				
			} catch (InterruptedException e) {
				e.printStackTrace();
			} finally {
				receiverLock.unlock();
			}
		}
		return signal;
	}
}
