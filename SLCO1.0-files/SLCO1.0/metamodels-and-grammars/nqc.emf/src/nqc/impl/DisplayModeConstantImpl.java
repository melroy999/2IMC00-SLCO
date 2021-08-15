/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.DisplayModeConstant;
import nqc.DisplayModeEnum;
import nqc.NqcPackage;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Display Mode Constant</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link nqc.impl.DisplayModeConstantImpl#getDisplayMode <em>Display Mode</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class DisplayModeConstantImpl extends ConstantExpressionImpl implements DisplayModeConstant {
	/**
	 * The default value of the '{@link #getDisplayMode() <em>Display Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDisplayMode()
	 * @generated
	 * @ordered
	 */
	protected static final DisplayModeEnum DISPLAY_MODE_EDEFAULT = DisplayModeEnum.DISPLAY_WATCH;

	/**
	 * The cached value of the '{@link #getDisplayMode() <em>Display Mode</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDisplayMode()
	 * @generated
	 * @ordered
	 */
	protected DisplayModeEnum displayMode = DISPLAY_MODE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected DisplayModeConstantImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getDisplayModeConstant();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DisplayModeEnum getDisplayMode() {
		return displayMode;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setDisplayMode(DisplayModeEnum newDisplayMode) {
		DisplayModeEnum oldDisplayMode = displayMode;
		displayMode = newDisplayMode == null ? DISPLAY_MODE_EDEFAULT : newDisplayMode;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, NqcPackage.DISPLAY_MODE_CONSTANT__DISPLAY_MODE, oldDisplayMode, displayMode));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case NqcPackage.DISPLAY_MODE_CONSTANT__DISPLAY_MODE:
				return getDisplayMode();
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
			case NqcPackage.DISPLAY_MODE_CONSTANT__DISPLAY_MODE:
				setDisplayMode((DisplayModeEnum)newValue);
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
			case NqcPackage.DISPLAY_MODE_CONSTANT__DISPLAY_MODE:
				setDisplayMode(DISPLAY_MODE_EDEFAULT);
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
			case NqcPackage.DISPLAY_MODE_CONSTANT__DISPLAY_MODE:
				return displayMode != DISPLAY_MODE_EDEFAULT;
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
		result.append(" (DisplayMode: ");
		result.append(displayMode);
		result.append(')');
		return result.toString();
	}

} //DisplayModeConstantImpl
