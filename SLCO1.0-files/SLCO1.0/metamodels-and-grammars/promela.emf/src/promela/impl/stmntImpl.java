/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import promela.PromelaPackage;
import promela.stmnt;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>stmnt</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.stmntImpl#getLabel <em>Label</em>}</li>
 *   <li>{@link promela.impl.stmntImpl#getUnless <em>Unless</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public abstract class stmntImpl extends stepImpl implements stmnt {
	/**
	 * The default value of the '{@link #getLabel() <em>Label</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getLabel()
	 * @generated
	 * @ordered
	 */
	protected static final String LABEL_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getLabel() <em>Label</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getLabel()
	 * @generated
	 * @ordered
	 */
	protected String label = LABEL_EDEFAULT;

	/**
	 * The cached value of the '{@link #getUnless() <em>Unless</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUnless()
	 * @generated
	 * @ordered
	 */
	protected stmnt unless;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected stmntImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.STMNT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getLabel() {
		return label;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setLabel(String newLabel) {
		String oldLabel = label;
		label = newLabel;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.STMNT__LABEL, oldLabel, label));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public stmnt getUnless() {
		return unless;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetUnless(stmnt newUnless, NotificationChain msgs) {
		stmnt oldUnless = unless;
		unless = newUnless;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.STMNT__UNLESS, oldUnless, newUnless);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setUnless(stmnt newUnless) {
		if (newUnless != unless) {
			NotificationChain msgs = null;
			if (unless != null)
				msgs = ((InternalEObject)unless).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.STMNT__UNLESS, null, msgs);
			if (newUnless != null)
				msgs = ((InternalEObject)newUnless).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.STMNT__UNLESS, null, msgs);
			msgs = basicSetUnless(newUnless, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.STMNT__UNLESS, newUnless, newUnless));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.STMNT__UNLESS:
				return basicSetUnless(null, msgs);
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
			case PromelaPackage.STMNT__LABEL:
				return getLabel();
			case PromelaPackage.STMNT__UNLESS:
				return getUnless();
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
			case PromelaPackage.STMNT__LABEL:
				setLabel((String)newValue);
				return;
			case PromelaPackage.STMNT__UNLESS:
				setUnless((stmnt)newValue);
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
			case PromelaPackage.STMNT__LABEL:
				setLabel(LABEL_EDEFAULT);
				return;
			case PromelaPackage.STMNT__UNLESS:
				setUnless((stmnt)null);
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
			case PromelaPackage.STMNT__LABEL:
				return LABEL_EDEFAULT == null ? label != null : !LABEL_EDEFAULT.equals(label);
			case PromelaPackage.STMNT__UNLESS:
				return unless != null;
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
		result.append(" (label: ");
		result.append(label);
		result.append(')');
		return result.toString();
	}

} //stmntImpl
