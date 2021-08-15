/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import promela.PromelaPackage;
import promela.c_code;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>ccode</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.c_codeImpl#getEmbedded_c_code <em>Embedded ccode</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class c_codeImpl extends stmntImpl implements c_code {
	/**
	 * The default value of the '{@link #getEmbedded_c_code() <em>Embedded ccode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getEmbedded_c_code()
	 * @generated
	 * @ordered
	 */
	protected static final String EMBEDDED_CCODE_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getEmbedded_c_code() <em>Embedded ccode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getEmbedded_c_code()
	 * @generated
	 * @ordered
	 */
	protected String embedded_c_code = EMBEDDED_CCODE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected c_codeImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.CCODE;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getEmbedded_c_code() {
		return embedded_c_code;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setEmbedded_c_code(String newEmbedded_c_code) {
		String oldEmbedded_c_code = embedded_c_code;
		embedded_c_code = newEmbedded_c_code;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.CCODE__EMBEDDED_CCODE, oldEmbedded_c_code, embedded_c_code));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.CCODE__EMBEDDED_CCODE:
				return getEmbedded_c_code();
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
			case PromelaPackage.CCODE__EMBEDDED_CCODE:
				setEmbedded_c_code((String)newValue);
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
			case PromelaPackage.CCODE__EMBEDDED_CCODE:
				setEmbedded_c_code(EMBEDDED_CCODE_EDEFAULT);
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
			case PromelaPackage.CCODE__EMBEDDED_CCODE:
				return EMBEDDED_CCODE_EDEFAULT == null ? embedded_c_code != null : !EMBEDDED_CCODE_EDEFAULT.equals(embedded_c_code);
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
		result.append(" (embedded_c_code: ");
		result.append(embedded_c_code);
		result.append(')');
		return result.toString();
	}

} //c_codeImpl
