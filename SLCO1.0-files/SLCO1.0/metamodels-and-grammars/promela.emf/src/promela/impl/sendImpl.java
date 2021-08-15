/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import promela.PromelaPackage;
import promela.any_expr;
import promela.send;
import promela.sendtypeenum;
import promela.varref;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>send</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.sendImpl#getVarref <em>Varref</em>}</li>
 *   <li>{@link promela.impl.sendImpl#getSend_args <em>Send args</em>}</li>
 *   <li>{@link promela.impl.sendImpl#getSendtype <em>Sendtype</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class sendImpl extends stmntImpl implements send {
	/**
	 * The cached value of the '{@link #getVarref() <em>Varref</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVarref()
	 * @generated
	 * @ordered
	 */
	protected varref varref;

	/**
	 * The cached value of the '{@link #getSend_args() <em>Send args</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSend_args()
	 * @generated
	 * @ordered
	 */
	protected EList<any_expr> send_args;

	/**
	 * The default value of the '{@link #getSendtype() <em>Sendtype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSendtype()
	 * @generated
	 * @ordered
	 */
	protected static final sendtypeenum SENDTYPE_EDEFAULT = sendtypeenum.NORMAL_FIFO;

	/**
	 * The cached value of the '{@link #getSendtype() <em>Sendtype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSendtype()
	 * @generated
	 * @ordered
	 */
	protected sendtypeenum sendtype = SENDTYPE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected sendImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.SEND;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public varref getVarref() {
		return varref;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetVarref(varref newVarref, NotificationChain msgs) {
		varref oldVarref = varref;
		varref = newVarref;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.SEND__VARREF, oldVarref, newVarref);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setVarref(varref newVarref) {
		if (newVarref != varref) {
			NotificationChain msgs = null;
			if (varref != null)
				msgs = ((InternalEObject)varref).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.SEND__VARREF, null, msgs);
			if (newVarref != null)
				msgs = ((InternalEObject)newVarref).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.SEND__VARREF, null, msgs);
			msgs = basicSetVarref(newVarref, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.SEND__VARREF, newVarref, newVarref));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<any_expr> getSend_args() {
		if (send_args == null) {
			send_args = new EObjectContainmentEList<any_expr>(any_expr.class, this, PromelaPackage.SEND__SEND_ARGS);
		}
		return send_args;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public sendtypeenum getSendtype() {
		return sendtype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSendtype(sendtypeenum newSendtype) {
		sendtypeenum oldSendtype = sendtype;
		sendtype = newSendtype == null ? SENDTYPE_EDEFAULT : newSendtype;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.SEND__SENDTYPE, oldSendtype, sendtype));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.SEND__VARREF:
				return basicSetVarref(null, msgs);
			case PromelaPackage.SEND__SEND_ARGS:
				return ((InternalEList<?>)getSend_args()).basicRemove(otherEnd, msgs);
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
			case PromelaPackage.SEND__VARREF:
				return getVarref();
			case PromelaPackage.SEND__SEND_ARGS:
				return getSend_args();
			case PromelaPackage.SEND__SENDTYPE:
				return getSendtype();
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
			case PromelaPackage.SEND__VARREF:
				setVarref((varref)newValue);
				return;
			case PromelaPackage.SEND__SEND_ARGS:
				getSend_args().clear();
				getSend_args().addAll((Collection<? extends any_expr>)newValue);
				return;
			case PromelaPackage.SEND__SENDTYPE:
				setSendtype((sendtypeenum)newValue);
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
			case PromelaPackage.SEND__VARREF:
				setVarref((varref)null);
				return;
			case PromelaPackage.SEND__SEND_ARGS:
				getSend_args().clear();
				return;
			case PromelaPackage.SEND__SENDTYPE:
				setSendtype(SENDTYPE_EDEFAULT);
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
			case PromelaPackage.SEND__VARREF:
				return varref != null;
			case PromelaPackage.SEND__SEND_ARGS:
				return send_args != null && !send_args.isEmpty();
			case PromelaPackage.SEND__SENDTYPE:
				return sendtype != SENDTYPE_EDEFAULT;
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
		result.append(" (sendtype: ");
		result.append(sendtype);
		result.append(')');
		return result.toString();
	}

} //sendImpl
