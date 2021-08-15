/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.EObjectResolvingEList;
import org.eclipse.emf.ecore.util.InternalEList;

import poosl.Expression;
import poosl.IncomingMessage;
import poosl.MessageReceive;
import poosl.PooslPackage;
import poosl.Port;
import poosl.Variable;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Message Receive</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.MessageReceiveImpl#getMessage <em>Message</em>}</li>
 *   <li>{@link poosl.impl.MessageReceiveImpl#getPort <em>Port</em>}</li>
 *   <li>{@link poosl.impl.MessageReceiveImpl#getVariables <em>Variables</em>}</li>
 *   <li>{@link poosl.impl.MessageReceiveImpl#getCondition <em>Condition</em>}</li>
 *   <li>{@link poosl.impl.MessageReceiveImpl#getExpressions <em>Expressions</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class MessageReceiveImpl extends StatementImpl implements MessageReceive {
	/**
	 * The cached value of the '{@link #getMessage() <em>Message</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMessage()
	 * @generated
	 * @ordered
	 */
	protected IncomingMessage message;

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
	 * The cached value of the '{@link #getVariables() <em>Variables</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVariables()
	 * @generated
	 * @ordered
	 */
	protected EList<Variable> variables;

	/**
	 * The cached value of the '{@link #getCondition() <em>Condition</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCondition()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> condition;

	/**
	 * The cached value of the '{@link #getExpressions() <em>Expressions</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getExpressions()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> expressions;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MessageReceiveImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.MESSAGE_RECEIVE;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IncomingMessage getMessage() {
		return message;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetMessage(IncomingMessage newMessage, NotificationChain msgs) {
		IncomingMessage oldMessage = message;
		message = newMessage;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PooslPackage.MESSAGE_RECEIVE__MESSAGE, oldMessage, newMessage);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMessage(IncomingMessage newMessage) {
		if (newMessage != message) {
			NotificationChain msgs = null;
			if (message != null)
				msgs = ((InternalEObject)message).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PooslPackage.MESSAGE_RECEIVE__MESSAGE, null, msgs);
			if (newMessage != null)
				msgs = ((InternalEObject)newMessage).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PooslPackage.MESSAGE_RECEIVE__MESSAGE, null, msgs);
			msgs = basicSetMessage(newMessage, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.MESSAGE_RECEIVE__MESSAGE, newMessage, newMessage));
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
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.MESSAGE_RECEIVE__PORT, oldPort, port));
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
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.MESSAGE_RECEIVE__PORT, oldPort, port));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Variable> getVariables() {
		if (variables == null) {
			variables = new EObjectResolvingEList<Variable>(Variable.class, this, PooslPackage.MESSAGE_RECEIVE__VARIABLES);
		}
		return variables;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getCondition() {
		if (condition == null) {
			condition = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.MESSAGE_RECEIVE__CONDITION);
		}
		return condition;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getExpressions() {
		if (expressions == null) {
			expressions = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.MESSAGE_RECEIVE__EXPRESSIONS);
		}
		return expressions;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.MESSAGE_RECEIVE__MESSAGE:
				return basicSetMessage(null, msgs);
			case PooslPackage.MESSAGE_RECEIVE__CONDITION:
				return ((InternalEList<?>)getCondition()).basicRemove(otherEnd, msgs);
			case PooslPackage.MESSAGE_RECEIVE__EXPRESSIONS:
				return ((InternalEList<?>)getExpressions()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.MESSAGE_RECEIVE__MESSAGE:
				return getMessage();
			case PooslPackage.MESSAGE_RECEIVE__PORT:
				if (resolve) return getPort();
				return basicGetPort();
			case PooslPackage.MESSAGE_RECEIVE__VARIABLES:
				return getVariables();
			case PooslPackage.MESSAGE_RECEIVE__CONDITION:
				return getCondition();
			case PooslPackage.MESSAGE_RECEIVE__EXPRESSIONS:
				return getExpressions();
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
			case PooslPackage.MESSAGE_RECEIVE__MESSAGE:
				setMessage((IncomingMessage)newValue);
				return;
			case PooslPackage.MESSAGE_RECEIVE__PORT:
				setPort((Port)newValue);
				return;
			case PooslPackage.MESSAGE_RECEIVE__VARIABLES:
				getVariables().clear();
				getVariables().addAll((Collection<? extends Variable>)newValue);
				return;
			case PooslPackage.MESSAGE_RECEIVE__CONDITION:
				getCondition().clear();
				getCondition().addAll((Collection<? extends Expression>)newValue);
				return;
			case PooslPackage.MESSAGE_RECEIVE__EXPRESSIONS:
				getExpressions().clear();
				getExpressions().addAll((Collection<? extends Expression>)newValue);
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
			case PooslPackage.MESSAGE_RECEIVE__MESSAGE:
				setMessage((IncomingMessage)null);
				return;
			case PooslPackage.MESSAGE_RECEIVE__PORT:
				setPort((Port)null);
				return;
			case PooslPackage.MESSAGE_RECEIVE__VARIABLES:
				getVariables().clear();
				return;
			case PooslPackage.MESSAGE_RECEIVE__CONDITION:
				getCondition().clear();
				return;
			case PooslPackage.MESSAGE_RECEIVE__EXPRESSIONS:
				getExpressions().clear();
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
			case PooslPackage.MESSAGE_RECEIVE__MESSAGE:
				return message != null;
			case PooslPackage.MESSAGE_RECEIVE__PORT:
				return port != null;
			case PooslPackage.MESSAGE_RECEIVE__VARIABLES:
				return variables != null && !variables.isEmpty();
			case PooslPackage.MESSAGE_RECEIVE__CONDITION:
				return condition != null && !condition.isEmpty();
			case PooslPackage.MESSAGE_RECEIVE__EXPRESSIONS:
				return expressions != null && !expressions.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //MessageReceiveImpl
