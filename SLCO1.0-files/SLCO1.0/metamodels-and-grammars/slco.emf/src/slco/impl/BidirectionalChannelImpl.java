/**
 */
package slco.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import slco.BidirectionalChannel;
import slco.Port;
import slco.SlcoPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Bidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco.impl.BidirectionalChannelImpl#getPort1 <em>Port1</em>}</li>
 *   <li>{@link slco.impl.BidirectionalChannelImpl#getPort2 <em>Port2</em>}</li>
 *   <li>{@link slco.impl.BidirectionalChannelImpl#getObject1 <em>Object1</em>}</li>
 *   <li>{@link slco.impl.BidirectionalChannelImpl#getObject2 <em>Object2</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class BidirectionalChannelImpl extends ChannelImpl implements BidirectionalChannel {
	/**
	 * The cached value of the '{@link #getPort1() <em>Port1</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPort1()
	 * @generated
	 * @ordered
	 */
	protected Port port1;

	/**
	 * The cached value of the '{@link #getPort2() <em>Port2</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPort2()
	 * @generated
	 * @ordered
	 */
	protected Port port2;

	/**
	 * The cached value of the '{@link #getObject1() <em>Object1</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getObject1()
	 * @generated
	 * @ordered
	 */
	protected slco.Object object1;

	/**
	 * The cached value of the '{@link #getObject2() <em>Object2</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getObject2()
	 * @generated
	 * @ordered
	 */
	protected slco.Object object2;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected BidirectionalChannelImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return SlcoPackage.Literals.BIDIRECTIONAL_CHANNEL;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port getPort1() {
		if (port1 != null && port1.eIsProxy()) {
			InternalEObject oldPort1 = (InternalEObject)port1;
			port1 = (Port)eResolveProxy(oldPort1);
			if (port1 != oldPort1) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1, oldPort1, port1));
			}
		}
		return port1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port basicGetPort1() {
		return port1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPort1(Port newPort1) {
		Port oldPort1 = port1;
		port1 = newPort1;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1, oldPort1, port1));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port getPort2() {
		if (port2 != null && port2.eIsProxy()) {
			InternalEObject oldPort2 = (InternalEObject)port2;
			port2 = (Port)eResolveProxy(oldPort2);
			if (port2 != oldPort2) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2, oldPort2, port2));
			}
		}
		return port2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port basicGetPort2() {
		return port2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPort2(Port newPort2) {
		Port oldPort2 = port2;
		port2 = newPort2;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2, oldPort2, port2));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public slco.Object getObject1() {
		if (object1 != null && object1.eIsProxy()) {
			InternalEObject oldObject1 = (InternalEObject)object1;
			object1 = (slco.Object)eResolveProxy(oldObject1);
			if (object1 != oldObject1) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1, oldObject1, object1));
			}
		}
		return object1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public slco.Object basicGetObject1() {
		return object1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setObject1(slco.Object newObject1) {
		slco.Object oldObject1 = object1;
		object1 = newObject1;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1, oldObject1, object1));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public slco.Object getObject2() {
		if (object2 != null && object2.eIsProxy()) {
			InternalEObject oldObject2 = (InternalEObject)object2;
			object2 = (slco.Object)eResolveProxy(oldObject2);
			if (object2 != oldObject2) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2, oldObject2, object2));
			}
		}
		return object2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public slco.Object basicGetObject2() {
		return object2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setObject2(slco.Object newObject2) {
		slco.Object oldObject2 = object2;
		object2 = newObject2;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2, oldObject2, object2));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1:
				if (resolve) return getPort1();
				return basicGetPort1();
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2:
				if (resolve) return getPort2();
				return basicGetPort2();
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1:
				if (resolve) return getObject1();
				return basicGetObject1();
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2:
				if (resolve) return getObject2();
				return basicGetObject2();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1:
				setPort1((Port)newValue);
				return;
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2:
				setPort2((Port)newValue);
				return;
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1:
				setObject1((slco.Object)newValue);
				return;
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2:
				setObject2((slco.Object)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1:
				setPort1((Port)null);
				return;
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2:
				setPort2((Port)null);
				return;
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1:
				setObject1((slco.Object)null);
				return;
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2:
				setObject2((slco.Object)null);
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1:
				return port1 != null;
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2:
				return port2 != null;
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1:
				return object1 != null;
			case SlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2:
				return object2 != null;
		}
		return super.eIsSet(featureID);
	}

} //BidirectionalChannelImpl
