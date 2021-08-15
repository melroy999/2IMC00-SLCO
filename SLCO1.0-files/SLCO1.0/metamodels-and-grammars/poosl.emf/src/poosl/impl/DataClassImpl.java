/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import poosl.DataClass;
import poosl.DataMethod;
import poosl.PooslPackage;
import poosl.Variable;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Data Class</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.DataClassImpl#getVariables <em>Variables</em>}</li>
 *   <li>{@link poosl.impl.DataClassImpl#getMethods <em>Methods</em>}</li>
 *   <li>{@link poosl.impl.DataClassImpl#getSuperClass <em>Super Class</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class DataClassImpl extends ClassImpl implements DataClass {
	/**
	 * The cached value of the '{@link #getVariables() <em>Variables</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVariables()
	 * @generated
	 * @ordered
	 */
	protected EList<Variable> variables;

	/**
	 * The cached value of the '{@link #getMethods() <em>Methods</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMethods()
	 * @generated
	 * @ordered
	 */
	protected EList<DataMethod> methods;

	/**
	 * The cached value of the '{@link #getSuperClass() <em>Super Class</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSuperClass()
	 * @generated
	 * @ordered
	 */
	protected DataClass superClass;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected DataClassImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.DATA_CLASS;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Variable> getVariables() {
		if (variables == null) {
			variables = new EObjectContainmentEList<Variable>(Variable.class, this, PooslPackage.DATA_CLASS__VARIABLES);
		}
		return variables;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<DataMethod> getMethods() {
		if (methods == null) {
			methods = new EObjectContainmentEList<DataMethod>(DataMethod.class, this, PooslPackage.DATA_CLASS__METHODS);
		}
		return methods;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataClass getSuperClass() {
		if (superClass != null && superClass.eIsProxy()) {
			InternalEObject oldSuperClass = (InternalEObject)superClass;
			superClass = (DataClass)eResolveProxy(oldSuperClass);
			if (superClass != oldSuperClass) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.DATA_CLASS__SUPER_CLASS, oldSuperClass, superClass));
			}
		}
		return superClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataClass basicGetSuperClass() {
		return superClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSuperClass(DataClass newSuperClass) {
		DataClass oldSuperClass = superClass;
		superClass = newSuperClass;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.DATA_CLASS__SUPER_CLASS, oldSuperClass, superClass));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.DATA_CLASS__VARIABLES:
				return ((InternalEList<?>)getVariables()).basicRemove(otherEnd, msgs);
			case PooslPackage.DATA_CLASS__METHODS:
				return ((InternalEList<?>)getMethods()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.DATA_CLASS__VARIABLES:
				return getVariables();
			case PooslPackage.DATA_CLASS__METHODS:
				return getMethods();
			case PooslPackage.DATA_CLASS__SUPER_CLASS:
				if (resolve) return getSuperClass();
				return basicGetSuperClass();
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
			case PooslPackage.DATA_CLASS__VARIABLES:
				getVariables().clear();
				getVariables().addAll((Collection<? extends Variable>)newValue);
				return;
			case PooslPackage.DATA_CLASS__METHODS:
				getMethods().clear();
				getMethods().addAll((Collection<? extends DataMethod>)newValue);
				return;
			case PooslPackage.DATA_CLASS__SUPER_CLASS:
				setSuperClass((DataClass)newValue);
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
			case PooslPackage.DATA_CLASS__VARIABLES:
				getVariables().clear();
				return;
			case PooslPackage.DATA_CLASS__METHODS:
				getMethods().clear();
				return;
			case PooslPackage.DATA_CLASS__SUPER_CLASS:
				setSuperClass((DataClass)null);
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
			case PooslPackage.DATA_CLASS__VARIABLES:
				return variables != null && !variables.isEmpty();
			case PooslPackage.DATA_CLASS__METHODS:
				return methods != null && !methods.isEmpty();
			case PooslPackage.DATA_CLASS__SUPER_CLASS:
				return superClass != null;
		}
		return super.eIsSet(featureID);
	}

} //DataClassImpl
