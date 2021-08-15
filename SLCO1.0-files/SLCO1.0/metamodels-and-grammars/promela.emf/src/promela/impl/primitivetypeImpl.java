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
import promela.primitivetype;
import promela.primitivetypeenum;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>primitivetype</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.primitivetypeImpl#getPrimitivetype <em>Primitivetype</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class primitivetypeImpl extends typenameImpl implements primitivetype {
	/**
	 * The default value of the '{@link #getPrimitivetype() <em>Primitivetype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPrimitivetype()
	 * @generated
	 * @ordered
	 */
	protected static final primitivetypeenum PRIMITIVETYPE_EDEFAULT = primitivetypeenum.BIT;

	/**
	 * The cached value of the '{@link #getPrimitivetype() <em>Primitivetype</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPrimitivetype()
	 * @generated
	 * @ordered
	 */
	protected primitivetypeenum primitivetype = PRIMITIVETYPE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected primitivetypeImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.PRIMITIVETYPE;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public primitivetypeenum getPrimitivetype() {
		return primitivetype;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPrimitivetype(primitivetypeenum newPrimitivetype) {
		primitivetypeenum oldPrimitivetype = primitivetype;
		primitivetype = newPrimitivetype == null ? PRIMITIVETYPE_EDEFAULT : newPrimitivetype;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.PRIMITIVETYPE__PRIMITIVETYPE, oldPrimitivetype, primitivetype));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.PRIMITIVETYPE__PRIMITIVETYPE:
				return getPrimitivetype();
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
			case PromelaPackage.PRIMITIVETYPE__PRIMITIVETYPE:
				setPrimitivetype((primitivetypeenum)newValue);
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
			case PromelaPackage.PRIMITIVETYPE__PRIMITIVETYPE:
				setPrimitivetype(PRIMITIVETYPE_EDEFAULT);
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
			case PromelaPackage.PRIMITIVETYPE__PRIMITIVETYPE:
				return primitivetype != PRIMITIVETYPE_EDEFAULT;
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
		result.append(" (primitivetype: ");
		result.append(primitivetype);
		result.append(')');
		return result.toString();
	}

} //primitivetypeImpl
