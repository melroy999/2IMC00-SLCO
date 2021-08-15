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
import org.eclipse.emf.ecore.impl.EObjectImpl;

import promela.PromelaPackage;
import promela.any_expr_or_ch_init;
import promela.ivar;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>ivar</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.ivarImpl#getName <em>Name</em>}</li>
 *   <li>{@link promela.impl.ivarImpl#getConstant <em>Constant</em>}</li>
 *   <li>{@link promela.impl.ivarImpl#getAny_expr_or_ch_init <em>Any expr or ch init</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ivarImpl extends EObjectImpl implements ivar {
	/**
	 * The default value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected static final String NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected String name = NAME_EDEFAULT;

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
	 * The cached value of the '{@link #getAny_expr_or_ch_init() <em>Any expr or ch init</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAny_expr_or_ch_init()
	 * @generated
	 * @ordered
	 */
	protected any_expr_or_ch_init any_expr_or_ch_init;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ivarImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.IVAR;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
		return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.IVAR__NAME, oldName, name));
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
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.IVAR__CONSTANT, oldConstant, constant));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public any_expr_or_ch_init getAny_expr_or_ch_init() {
		return any_expr_or_ch_init;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetAny_expr_or_ch_init(any_expr_or_ch_init newAny_expr_or_ch_init, NotificationChain msgs) {
		any_expr_or_ch_init oldAny_expr_or_ch_init = any_expr_or_ch_init;
		any_expr_or_ch_init = newAny_expr_or_ch_init;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT, oldAny_expr_or_ch_init, newAny_expr_or_ch_init);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAny_expr_or_ch_init(any_expr_or_ch_init newAny_expr_or_ch_init) {
		if (newAny_expr_or_ch_init != any_expr_or_ch_init) {
			NotificationChain msgs = null;
			if (any_expr_or_ch_init != null)
				msgs = ((InternalEObject)any_expr_or_ch_init).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT, null, msgs);
			if (newAny_expr_or_ch_init != null)
				msgs = ((InternalEObject)newAny_expr_or_ch_init).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT, null, msgs);
			msgs = basicSetAny_expr_or_ch_init(newAny_expr_or_ch_init, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT, newAny_expr_or_ch_init, newAny_expr_or_ch_init));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT:
				return basicSetAny_expr_or_ch_init(null, msgs);
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
			case PromelaPackage.IVAR__NAME:
				return getName();
			case PromelaPackage.IVAR__CONSTANT:
				return getConstant();
			case PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT:
				return getAny_expr_or_ch_init();
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
			case PromelaPackage.IVAR__NAME:
				setName((String)newValue);
				return;
			case PromelaPackage.IVAR__CONSTANT:
				setConstant((Integer)newValue);
				return;
			case PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT:
				setAny_expr_or_ch_init((any_expr_or_ch_init)newValue);
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
			case PromelaPackage.IVAR__NAME:
				setName(NAME_EDEFAULT);
				return;
			case PromelaPackage.IVAR__CONSTANT:
				setConstant(CONSTANT_EDEFAULT);
				return;
			case PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT:
				setAny_expr_or_ch_init((any_expr_or_ch_init)null);
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
			case PromelaPackage.IVAR__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case PromelaPackage.IVAR__CONSTANT:
				return constant != CONSTANT_EDEFAULT;
			case PromelaPackage.IVAR__ANY_EXPR_OR_CH_INIT:
				return any_expr_or_ch_init != null;
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
		result.append(" (name: ");
		result.append(name);
		result.append(", constant: ");
		result.append(constant);
		result.append(')');
		return result.toString();
	}

} //ivarImpl
