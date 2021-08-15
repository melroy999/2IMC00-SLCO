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
import promela.receive;
import promela.receivetypeenum;
import promela.recv_arg;
import promela.varref;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>receive</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.receiveImpl#getVarref <em>Varref</em>}</li>
 *   <li>{@link promela.impl.receiveImpl#getRecv_args <em>Recv args</em>}</li>
 *   <li>{@link promela.impl.receiveImpl#getReceivetype <em>Receivetype</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class receiveImpl extends stmntImpl implements receive {
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
	 * The default value of the '{@link #getReceivetype() <em>Receivetype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReceivetype()
	 * @generated
	 * @ordered
	 */
	protected static final receivetypeenum RECEIVETYPE_EDEFAULT = receivetypeenum.NORMAL;

	/**
	 * The cached value of the '{@link #getReceivetype() <em>Receivetype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReceivetype()
	 * @generated
	 * @ordered
	 */
	protected receivetypeenum receivetype = RECEIVETYPE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected receiveImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.RECEIVE;
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.RECEIVE__VARREF, oldVarref, newVarref);
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
				msgs = ((InternalEObject)varref).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.RECEIVE__VARREF, null, msgs);
			if (newVarref != null)
				msgs = ((InternalEObject)newVarref).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.RECEIVE__VARREF, null, msgs);
			msgs = basicSetVarref(newVarref, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.RECEIVE__VARREF, newVarref, newVarref));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<recv_arg> getRecv_args() {
		if (recv_args == null) {
			recv_args = new EObjectContainmentEList<recv_arg>(recv_arg.class, this, PromelaPackage.RECEIVE__RECV_ARGS);
		}
		return recv_args;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public receivetypeenum getReceivetype() {
		return receivetype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setReceivetype(receivetypeenum newReceivetype) {
		receivetypeenum oldReceivetype = receivetype;
		receivetype = newReceivetype == null ? RECEIVETYPE_EDEFAULT : newReceivetype;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.RECEIVE__RECEIVETYPE, oldReceivetype, receivetype));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.RECEIVE__VARREF:
				return basicSetVarref(null, msgs);
			case PromelaPackage.RECEIVE__RECV_ARGS:
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
			case PromelaPackage.RECEIVE__VARREF:
				return getVarref();
			case PromelaPackage.RECEIVE__RECV_ARGS:
				return getRecv_args();
			case PromelaPackage.RECEIVE__RECEIVETYPE:
				return getReceivetype();
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
			case PromelaPackage.RECEIVE__VARREF:
				setVarref((varref)newValue);
				return;
			case PromelaPackage.RECEIVE__RECV_ARGS:
				getRecv_args().clear();
				getRecv_args().addAll((Collection<? extends recv_arg>)newValue);
				return;
			case PromelaPackage.RECEIVE__RECEIVETYPE:
				setReceivetype((receivetypeenum)newValue);
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
			case PromelaPackage.RECEIVE__VARREF:
				setVarref((varref)null);
				return;
			case PromelaPackage.RECEIVE__RECV_ARGS:
				getRecv_args().clear();
				return;
			case PromelaPackage.RECEIVE__RECEIVETYPE:
				setReceivetype(RECEIVETYPE_EDEFAULT);
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
			case PromelaPackage.RECEIVE__VARREF:
				return varref != null;
			case PromelaPackage.RECEIVE__RECV_ARGS:
				return recv_args != null && !recv_args.isEmpty();
			case PromelaPackage.RECEIVE__RECEIVETYPE:
				return receivetype != RECEIVETYPE_EDEFAULT;
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
		result.append(" (receivetype: ");
		result.append(receivetype);
		result.append(')');
		return result.toString();
	}

} //receiveImpl
