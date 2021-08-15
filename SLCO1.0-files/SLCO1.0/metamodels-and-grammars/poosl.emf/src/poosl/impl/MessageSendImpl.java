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
import org.eclipse.emf.ecore.util.InternalEList;

import poosl.Expression;
import poosl.ExpressionList;
import poosl.MessageSend;
import poosl.OutgoingMessage;
import poosl.PooslPackage;
import poosl.Port;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Message Send</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.MessageSendImpl#getMessage <em>Message</em>}</li>
 *   <li>{@link poosl.impl.MessageSendImpl#getPort <em>Port</em>}</li>
 *   <li>{@link poosl.impl.MessageSendImpl#getExpressions <em>Expressions</em>}</li>
 *   <li>{@link poosl.impl.MessageSendImpl#getExpressionLists <em>Expression Lists</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class MessageSendImpl extends StatementImpl implements MessageSend {
	/**
	 * The cached value of the '{@link #getMessage() <em>Message</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMessage()
	 * @generated
	 * @ordered
	 */
	protected OutgoingMessage message;

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
	 * The cached value of the '{@link #getExpressions() <em>Expressions</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getExpressions()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> expressions;

	/**
	 * The cached value of the '{@link #getExpressionLists() <em>Expression Lists</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getExpressionLists()
	 * @generated
	 * @ordered
	 */
	protected EList<ExpressionList> expressionLists;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MessageSendImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.MESSAGE_SEND;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public OutgoingMessage getMessage() {
		return message;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetMessage(OutgoingMessage newMessage, NotificationChain msgs) {
		OutgoingMessage oldMessage = message;
		message = newMessage;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PooslPackage.MESSAGE_SEND__MESSAGE, oldMessage, newMessage);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMessage(OutgoingMessage newMessage) {
		if (newMessage != message) {
			NotificationChain msgs = null;
			if (message != null)
				msgs = ((InternalEObject)message).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PooslPackage.MESSAGE_SEND__MESSAGE, null, msgs);
			if (newMessage != null)
				msgs = ((InternalEObject)newMessage).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PooslPackage.MESSAGE_SEND__MESSAGE, null, msgs);
			msgs = basicSetMessage(newMessage, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.MESSAGE_SEND__MESSAGE, newMessage, newMessage));
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
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.MESSAGE_SEND__PORT, oldPort, port));
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
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.MESSAGE_SEND__PORT, oldPort, port));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getExpressions() {
		if (expressions == null) {
			expressions = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.MESSAGE_SEND__EXPRESSIONS);
		}
		return expressions;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ExpressionList> getExpressionLists() {
		if (expressionLists == null) {
			expressionLists = new EObjectContainmentEList<ExpressionList>(ExpressionList.class, this, PooslPackage.MESSAGE_SEND__EXPRESSION_LISTS);
		}
		return expressionLists;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.MESSAGE_SEND__MESSAGE:
				return basicSetMessage(null, msgs);
			case PooslPackage.MESSAGE_SEND__EXPRESSIONS:
				return ((InternalEList<?>)getExpressions()).basicRemove(otherEnd, msgs);
			case PooslPackage.MESSAGE_SEND__EXPRESSION_LISTS:
				return ((InternalEList<?>)getExpressionLists()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.MESSAGE_SEND__MESSAGE:
				return getMessage();
			case PooslPackage.MESSAGE_SEND__PORT:
				if (resolve) return getPort();
				return basicGetPort();
			case PooslPackage.MESSAGE_SEND__EXPRESSIONS:
				return getExpressions();
			case PooslPackage.MESSAGE_SEND__EXPRESSION_LISTS:
				return getExpressionLists();
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
			case PooslPackage.MESSAGE_SEND__MESSAGE:
				setMessage((OutgoingMessage)newValue);
				return;
			case PooslPackage.MESSAGE_SEND__PORT:
				setPort((Port)newValue);
				return;
			case PooslPackage.MESSAGE_SEND__EXPRESSIONS:
				getExpressions().clear();
				getExpressions().addAll((Collection<? extends Expression>)newValue);
				return;
			case PooslPackage.MESSAGE_SEND__EXPRESSION_LISTS:
				getExpressionLists().clear();
				getExpressionLists().addAll((Collection<? extends ExpressionList>)newValue);
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
			case PooslPackage.MESSAGE_SEND__MESSAGE:
				setMessage((OutgoingMessage)null);
				return;
			case PooslPackage.MESSAGE_SEND__PORT:
				setPort((Port)null);
				return;
			case PooslPackage.MESSAGE_SEND__EXPRESSIONS:
				getExpressions().clear();
				return;
			case PooslPackage.MESSAGE_SEND__EXPRESSION_LISTS:
				getExpressionLists().clear();
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
			case PooslPackage.MESSAGE_SEND__MESSAGE:
				return message != null;
			case PooslPackage.MESSAGE_SEND__PORT:
				return port != null;
			case PooslPackage.MESSAGE_SEND__EXPRESSIONS:
				return expressions != null && !expressions.isEmpty();
			case PooslPackage.MESSAGE_SEND__EXPRESSION_LISTS:
				return expressionLists != null && !expressionLists.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //MessageSendImpl
