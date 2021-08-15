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
import promela.c_track;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>ctrack</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.c_trackImpl#getEmbedded_c_code <em>Embedded ccode</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class c_trackImpl extends stmntImpl implements c_track {
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
	protected c_trackImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.CTRACK;
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
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.CTRACK__EMBEDDED_CCODE, oldEmbedded_c_code, embedded_c_code));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.CTRACK__EMBEDDED_CCODE:
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
			case PromelaPackage.CTRACK__EMBEDDED_CCODE:
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
			case PromelaPackage.CTRACK__EMBEDDED_CCODE:
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
			case PromelaPackage.CTRACK__EMBEDDED_CCODE:
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

} //c_trackImpl
