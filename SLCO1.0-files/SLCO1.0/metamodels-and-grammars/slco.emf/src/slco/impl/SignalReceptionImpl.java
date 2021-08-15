/**
 */
package slco.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import slco.Expression;
import slco.Port;
import slco.SignalArgument;
import slco.SignalReception;
import slco.SlcoPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Signal Reception</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco.impl.SignalReceptionImpl#getPort <em>Port</em>}</li>
 *   <li>{@link slco.impl.SignalReceptionImpl#getSignalName <em>Signal Name</em>}</li>
 *   <li>{@link slco.impl.SignalReceptionImpl#getArguments <em>Arguments</em>}</li>
 *   <li>{@link slco.impl.SignalReceptionImpl#getCondition <em>Condition</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SignalReceptionImpl extends StatementImpl implements SignalReception {
	/**
	 * The cached value of the '{@link #getPort() <em>Port</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPort()
	 * @generated
	 * @ordered
	 */
	protected Port port;

	/**
	 * The default value of the '{@link #getSignalName() <em>Signal Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignalName()
	 * @generated
	 * @ordered
	 */
	protected static final String SIGNAL_NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getSignalName() <em>Signal Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignalName()
	 * @generated
	 * @ordered
	 */
	protected String signalName = SIGNAL_NAME_EDEFAULT;

	/**
	 * The cached value of the '{@link #getArguments() <em>Arguments</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getArguments()
	 * @generated
	 * @ordered
	 */
	protected EList<SignalArgument> arguments;

	/**
	 * The cached value of the '{@link #getCondition() <em>Condition</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCondition()
	 * @generated
	 * @ordered
	 */
	protected Expression condition;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SignalReceptionImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return SlcoPackage.Literals.SIGNAL_RECEPTION;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port getPort() {
		if (port != null && port.eIsProxy()) {
			InternalEObject oldPort = (InternalEObject)port;
			port = (Port)eResolveProxy(oldPort);
			if (port != oldPort) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, SlcoPackage.SIGNAL_RECEPTION__PORT, oldPort, port));
			}
		}
		return port;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Port basicGetPort() {
		return port;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPort(Port newPort) {
		Port oldPort = port;
		port = newPort;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.SIGNAL_RECEPTION__PORT, oldPort, port));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getSignalName() {
		return signalName;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSignalName(String newSignalName) {
		String oldSignalName = signalName;
		signalName = newSignalName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.SIGNAL_RECEPTION__SIGNAL_NAME, oldSignalName, signalName));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<SignalArgument> getArguments() {
		if (arguments == null) {
			arguments = new EObjectContainmentEList<SignalArgument>(SignalArgument.class, this, SlcoPackage.SIGNAL_RECEPTION__ARGUMENTS);
		}
		return arguments;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Expression getCondition() {
		return condition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetCondition(Expression newCondition, NotificationChain msgs) {
		Expression oldCondition = condition;
		condition = newCondition;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, SlcoPackage.SIGNAL_RECEPTION__CONDITION, oldCondition, newCondition);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCondition(Expression newCondition) {
		if (newCondition != condition) {
			NotificationChain msgs = null;
			if (condition != null)
				msgs = ((InternalEObject)condition).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - SlcoPackage.SIGNAL_RECEPTION__CONDITION, null, msgs);
			if (newCondition != null)
				msgs = ((InternalEObject)newCondition).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - SlcoPackage.SIGNAL_RECEPTION__CONDITION, null, msgs);
			msgs = basicSetCondition(newCondition, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, SlcoPackage.SIGNAL_RECEPTION__CONDITION, newCondition, newCondition));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case SlcoPackage.SIGNAL_RECEPTION__ARGUMENTS:
				return ((InternalEList<?>)getArguments()).basicRemove(otherEnd, msgs);
			case SlcoPackage.SIGNAL_RECEPTION__CONDITION:
				return basicSetCondition(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case SlcoPackage.SIGNAL_RECEPTION__PORT:
				if (resolve) return getPort();
				return basicGetPort();
			case SlcoPackage.SIGNAL_RECEPTION__SIGNAL_NAME:
				return getSignalName();
			case SlcoPackage.SIGNAL_RECEPTION__ARGUMENTS:
				return getArguments();
			case SlcoPackage.SIGNAL_RECEPTION__CONDITION:
				return getCondition();
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
			case SlcoPackage.SIGNAL_RECEPTION__PORT:
				setPort((Port)newValue);
				return;
			case SlcoPackage.SIGNAL_RECEPTION__SIGNAL_NAME:
				setSignalName((String)newValue);
				return;
			case SlcoPackage.SIGNAL_RECEPTION__ARGUMENTS:
				getArguments().clear();
				getArguments().addAll((Collection<? extends SignalArgument>)newValue);
				return;
			case SlcoPackage.SIGNAL_RECEPTION__CONDITION:
				setCondition((Expression)newValue);
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
			case SlcoPackage.SIGNAL_RECEPTION__PORT:
				setPort((Port)null);
				return;
			case SlcoPackage.SIGNAL_RECEPTION__SIGNAL_NAME:
				setSignalName(SIGNAL_NAME_EDEFAULT);
				return;
			case SlcoPackage.SIGNAL_RECEPTION__ARGUMENTS:
				getArguments().clear();
				return;
			case SlcoPackage.SIGNAL_RECEPTION__CONDITION:
				setCondition((Expression)null);
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
			case SlcoPackage.SIGNAL_RECEPTION__PORT:
				return port != null;
			case SlcoPackage.SIGNAL_RECEPTION__SIGNAL_NAME:
				return SIGNAL_NAME_EDEFAULT == null ? signalName != null : !SIGNAL_NAME_EDEFAULT.equals(signalName);
			case SlcoPackage.SIGNAL_RECEPTION__ARGUMENTS:
				return arguments != null && !arguments.isEmpty();
			case SlcoPackage.SIGNAL_RECEPTION__CONDITION:
				return condition != null;
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
		result.append(" (signalName: ");
		result.append(signalName);
		result.append(')');
		return result.toString();
	}

} //SignalReceptionImpl
