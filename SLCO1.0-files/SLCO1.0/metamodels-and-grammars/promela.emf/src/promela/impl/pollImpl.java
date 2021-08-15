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
import promela.poll;
import promela.polltypeenum;
import promela.recv_arg;
import promela.varref;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>poll</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.pollImpl#getVarref <em>Varref</em>}</li>
 *   <li>{@link promela.impl.pollImpl#getRecv_args <em>Recv args</em>}</li>
 *   <li>{@link promela.impl.pollImpl#getPolltype <em>Polltype</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public abstract class pollImpl extends any_exprImpl implements poll {
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
	 * The cached value of the '{@link #getRecv_args() <em>Recv args</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getRecv_args()
	 * @generated
	 * @ordered
	 */
	protected EList<recv_arg> recv_args;

	/**
	 * The default value of the '{@link #getPolltype() <em>Polltype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPolltype()
	 * @generated
	 * @ordered
	 */
	protected static final polltypeenum POLLTYPE_EDEFAULT = polltypeenum.NORMAL_POLL_NO_SIDE_EFFECT;

	/**
	 * The cached value of the '{@link #getPolltype() <em>Polltype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPolltype()
	 * @generated
	 * @ordered
	 */
	protected polltypeenum polltype = POLLTYPE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected pollImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.POLL;
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.POLL__VARREF, oldVarref, newVarref);
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
				msgs = ((InternalEObject)varref).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.POLL__VARREF, null, msgs);
			if (newVarref != null)
				msgs = ((InternalEObject)newVarref).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.POLL__VARREF, null, msgs);
			msgs = basicSetVarref(newVarref, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.POLL__VARREF, newVarref, newVarref));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<recv_arg> getRecv_args() {
		if (recv_args == null) {
			recv_args = new EObjectContainmentEList<recv_arg>(recv_arg.class, this, PromelaPackage.POLL__RECV_ARGS);
		}
		return recv_args;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public polltypeenum getPolltype() {
		return polltype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPolltype(polltypeenum newPolltype) {
		polltypeenum oldPolltype = polltype;
		polltype = newPolltype == null ? POLLTYPE_EDEFAULT : newPolltype;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.POLL__POLLTYPE, oldPolltype, polltype));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.POLL__VARREF:
				return basicSetVarref(null, msgs);
			case PromelaPackage.POLL__RECV_ARGS:
				return ((InternalEList<?>)getRecv_args()).basicRemove(otherEnd, msgs);
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
			case PromelaPackage.POLL__VARREF:
				return getVarref();
			case PromelaPackage.POLL__RECV_ARGS:
				return getRecv_args();
			case PromelaPackage.POLL__POLLTYPE:
				return getPolltype();
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
			case PromelaPackage.POLL__VARREF:
				setVarref((varref)newValue);
				return;
			case PromelaPackage.POLL__RECV_ARGS:
				getRecv_args().clear();
				getRecv_args().addAll((Collection<? extends recv_arg>)newValue);
				return;
			case PromelaPackage.POLL__POLLTYPE:
				setPolltype((polltypeenum)newValue);
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
			case PromelaPackage.POLL__VARREF:
				setVarref((varref)null);
				return;
			case PromelaPackage.POLL__RECV_ARGS:
				getRecv_args().clear();
				return;
			case PromelaPackage.POLL__POLLTYPE:
				setPolltype(POLLTYPE_EDEFAULT);
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
			case PromelaPackage.POLL__VARREF:
				return varref != null;
			case PromelaPackage.POLL__RECV_ARGS:
				return recv_args != null && !recv_args.isEmpty();
			case PromelaPackage.POLL__POLLTYPE:
				return polltype != POLLTYPE_EDEFAULT;
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
		result.append(" (polltype: ");
		result.append(polltype);
		result.append(')');
		return result.toString();
	}

} //pollImpl
