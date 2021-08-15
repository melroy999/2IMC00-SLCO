/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectResolvingEList;

import poosl.Channel;
import poosl.PooslPackage;
import poosl.Port;
import poosl.ProcessInstance;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Channel</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.ChannelImpl#getName <em>Name</em>}</li>
 *   <li>{@link poosl.impl.ChannelImpl#getPort1 <em>Port1</em>}</li>
 *   <li>{@link poosl.impl.ChannelImpl#getPort2 <em>Port2</em>}</li>
 *   <li>{@link poosl.impl.ChannelImpl#getProcessInstance1 <em>Process Instance1</em>}</li>
 *   <li>{@link poosl.impl.ChannelImpl#getProcessInstance2 <em>Process Instance2</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ChannelImpl extends EObjectImpl implements Channel {
	/**
	 * The default value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected static final String NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected String name = NAME_EDEFAULT;

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
	 * The cached value of the '{@link #getProcessInstance1() <em>Process Instance1</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getProcessInstance1()
	 * @generated
	 * @ordered
	 */
	protected ProcessInstance processInstance1;

	/**
	 * The cached value of the '{@link #getProcessInstance2() <em>Process Instance2</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getProcessInstance2()
	 * @generated
	 * @ordered
	 */
	protected ProcessInstance processInstance2;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ChannelImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.CHANNEL;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
		return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CHANNEL__NAME, oldName, name));
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
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.CHANNEL__PORT1, oldPort1, port1));
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
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CHANNEL__PORT1, oldPort1, port1));
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
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.CHANNEL__PORT2, oldPort2, port2));
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
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CHANNEL__PORT2, oldPort2, port2));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessInstance getProcessInstance1() {
		if (processInstance1 != null && processInstance1.eIsProxy()) {
			InternalEObject oldProcessInstance1 = (InternalEObject)processInstance1;
			processInstance1 = (ProcessInstance)eResolveProxy(oldProcessInstance1);
			if (processInstance1 != oldProcessInstance1) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.CHANNEL__PROCESS_INSTANCE1, oldProcessInstance1, processInstance1));
			}
		}
		return processInstance1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessInstance basicGetProcessInstance1() {
		return processInstance1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setProcessInstance1(ProcessInstance newProcessInstance1) {
		ProcessInstance oldProcessInstance1 = processInstance1;
		processInstance1 = newProcessInstance1;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CHANNEL__PROCESS_INSTANCE1, oldProcessInstance1, processInstance1));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessInstance getProcessInstance2() {
		if (processInstance2 != null && processInstance2.eIsProxy()) {
			InternalEObject oldProcessInstance2 = (InternalEObject)processInstance2;
			processInstance2 = (ProcessInstance)eResolveProxy(oldProcessInstance2);
			if (processInstance2 != oldProcessInstance2) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.CHANNEL__PROCESS_INSTANCE2, oldProcessInstance2, processInstance2));
			}
		}
		return processInstance2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessInstance basicGetProcessInstance2() {
		return processInstance2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setProcessInstance2(ProcessInstance newProcessInstance2) {
		ProcessInstance oldProcessInstance2 = processInstance2;
		processInstance2 = newProcessInstance2;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.CHANNEL__PROCESS_INSTANCE2, oldProcessInstance2, processInstance2));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PooslPackage.CHANNEL__NAME:
				return getName();
			case PooslPackage.CHANNEL__PORT1:
				if (resolve) return getPort1();
				return basicGetPort1();
			case PooslPackage.CHANNEL__PORT2:
				if (resolve) return getPort2();
				return basicGetPort2();
			case PooslPackage.CHANNEL__PROCESS_INSTANCE1:
				if (resolve) return getProcessInstance1();
				return basicGetProcessInstance1();
			case PooslPackage.CHANNEL__PROCESS_INSTANCE2:
				if (resolve) return getProcessInstance2();
				return basicGetProcessInstance2();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case PooslPackage.CHANNEL__NAME:
				setName((String)newValue);
				return;
			case PooslPackage.CHANNEL__PORT1:
				setPort1((Port)newValue);
				return;
			case PooslPackage.CHANNEL__PORT2:
				setPort2((Port)newValue);
				return;
			case PooslPackage.CHANNEL__PROCESS_INSTANCE1:
				setProcessInstance1((ProcessInstance)newValue);
				return;
			case PooslPackage.CHANNEL__PROCESS_INSTANCE2:
				setProcessInstance2((ProcessInstance)newValue);
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
			case PooslPackage.CHANNEL__NAME:
				setName(NAME_EDEFAULT);
				return;
			case PooslPackage.CHANNEL__PORT1:
				setPort1((Port)null);
				return;
			case PooslPackage.CHANNEL__PORT2:
				setPort2((Port)null);
				return;
			case PooslPackage.CHANNEL__PROCESS_INSTANCE1:
				setProcessInstance1((ProcessInstance)null);
				return;
			case PooslPackage.CHANNEL__PROCESS_INSTANCE2:
				setProcessInstance2((ProcessInstance)null);
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
			case PooslPackage.CHANNEL__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case PooslPackage.CHANNEL__PORT1:
				return port1 != null;
			case PooslPackage.CHANNEL__PORT2:
				return port2 != null;
			case PooslPackage.CHANNEL__PROCESS_INSTANCE1:
				return processInstance1 != null;
			case PooslPackage.CHANNEL__PROCESS_INSTANCE2:
				return processInstance2 != null;
		}
		return super.eIsSet(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		if (eIsProxy()) return super.toString();

		StringBuffer result = new StringBuffer(super.toString());
		result.append(" (name: ");
		result.append(name);
		result.append(')');
		return result.toString();
	}

} //ChannelImpl
