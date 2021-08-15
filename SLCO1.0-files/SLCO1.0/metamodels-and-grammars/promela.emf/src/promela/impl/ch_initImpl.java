/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectResolvingEList;

import promela.PromelaPackage;
import promela.ch_init;
import promela.typename;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>ch init</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.ch_initImpl#getConstant <em>Constant</em>}</li>
 *   <li>{@link promela.impl.ch_initImpl#getTypenames <em>Typenames</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ch_initImpl extends any_expr_or_ch_initImpl implements ch_init {
	/**
	 * The default value of the '{@link #getConstant() <em>Constant</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getConstant()
	 * @generated
	 * @ordered
	 */
	protected static final int CONSTANT_EDEFAULT = 0;

	/**
	 * The cached value of the '{@link #getConstant() <em>Constant</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getConstant()
	 * @generated
	 * @ordered
	 */
	protected int constant = CONSTANT_EDEFAULT;

	/**
	 * The cached value of the '{@link #getTypenames() <em>Typenames</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTypenames()
	 * @generated
	 * @ordered
	 */
	protected EList<typename> typenames;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ch_initImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.CH_INIT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getConstant() {
		return constant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setConstant(int newConstant) {
		int oldConstant = constant;
		constant = newConstant;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.CH_INIT__CONSTANT, oldConstant, constant));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<typename> getTypenames() {
		if (typenames == null) {
			typenames = new EObjectResolvingEList<typename>(typename.class, this, PromelaPackage.CH_INIT__TYPENAMES);
		}
		return typenames;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.CH_INIT__CONSTANT:
				return getConstant();
			case PromelaPackage.CH_INIT__TYPENAMES:
				return getTypenames();
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
			case PromelaPackage.CH_INIT__CONSTANT:
				setConstant((Integer)newValue);
				return;
			case PromelaPackage.CH_INIT__TYPENAMES:
				getTypenames().clear();
				getTypenames().addAll((Collection<? extends typename>)newValue);
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
			case PromelaPackage.CH_INIT__CONSTANT:
				setConstant(CONSTANT_EDEFAULT);
				return;
			case PromelaPackage.CH_INIT__TYPENAMES:
				getTypenames().clear();
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
			case PromelaPackage.CH_INIT__CONSTANT:
				return constant != CONSTANT_EDEFAULT;
			case PromelaPackage.CH_INIT__TYPENAMES:
				return typenames != null && !typenames.isEmpty();
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
		result.append(" (constant: ");
		result.append(constant);
		result.append(')');
		return result.toString();
	}

} //ch_initImpl
