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

import poosl.DataMethod;
import poosl.DataMethodCall;
import poosl.Expression;
import poosl.ExpressionList;
import poosl.PooslPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Data Method Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.DataMethodCallImpl#isSuper <em>Super</em>}</li>
 *   <li>{@link poosl.impl.DataMethodCallImpl#getObjectExpressions <em>Object Expressions</em>}</li>
 *   <li>{@link poosl.impl.DataMethodCallImpl#getParameterExpressionLists <em>Parameter Expression Lists</em>}</li>
 *   <li>{@link poosl.impl.DataMethodCallImpl#getDataMethod <em>Data Method</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class DataMethodCallImpl extends ExpressionImpl implements DataMethodCall {
	/**
	 * The default value of the '{@link #isSuper() <em>Super</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isSuper()
	 * @generated
	 * @ordered
	 */
	protected static final boolean SUPER_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isSuper() <em>Super</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isSuper()
	 * @generated
	 * @ordered
	 */
	protected boolean super_ = SUPER_EDEFAULT;

	/**
	 * The cached value of the '{@link #getObjectExpressions() <em>Object Expressions</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getObjectExpressions()
	 * @generated
	 * @ordered
	 */
	protected EList<Expression> objectExpressions;

	/**
	 * The cached value of the '{@link #getParameterExpressionLists() <em>Parameter Expression Lists</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getParameterExpressionLists()
	 * @generated
	 * @ordered
	 */
	protected EList<ExpressionList> parameterExpressionLists;

	/**
	 * The cached value of the '{@link #getDataMethod() <em>Data Method</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDataMethod()
	 * @generated
	 * @ordered
	 */
	protected DataMethod dataMethod;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected DataMethodCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.DATA_METHOD_CALL;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public boolean isSuper() {
		return super_;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSuper(boolean newSuper) {
		boolean oldSuper = super_;
		super_ = newSuper;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.DATA_METHOD_CALL__SUPER, oldSuper, super_));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getObjectExpressions() {
		if (objectExpressions == null) {
			objectExpressions = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.DATA_METHOD_CALL__OBJECT_EXPRESSIONS);
		}
		return objectExpressions;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ExpressionList> getParameterExpressionLists() {
		if (parameterExpressionLists == null) {
			parameterExpressionLists = new EObjectContainmentEList<ExpressionList>(ExpressionList.class, this, PooslPackage.DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS);
		}
		return parameterExpressionLists;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataMethod getDataMethod() {
		if (dataMethod != null && dataMethod.eIsProxy()) {
			InternalEObject oldDataMethod = (InternalEObject)dataMethod;
			dataMethod = (DataMethod)eResolveProxy(oldDataMethod);
			if (dataMethod != oldDataMethod) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.DATA_METHOD_CALL__DATA_METHOD, oldDataMethod, dataMethod));
			}
		}
		return dataMethod;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public DataMethod basicGetDataMethod() {
		return dataMethod;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setDataMethod(DataMethod newDataMethod) {
		DataMethod oldDataMethod = dataMethod;
		dataMethod = newDataMethod;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.DATA_METHOD_CALL__DATA_METHOD, oldDataMethod, dataMethod));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				return ((InternalEList<?>)getObjectExpressions()).basicRemove(otherEnd, msgs);
			case PooslPackage.DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
				return ((InternalEList<?>)getParameterExpressionLists()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.DATA_METHOD_CALL__SUPER:
				return isSuper();
			case PooslPackage.DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				return getObjectExpressions();
			case PooslPackage.DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
				return getParameterExpressionLists();
			case PooslPackage.DATA_METHOD_CALL__DATA_METHOD:
				if (resolve) return getDataMethod();
				return basicGetDataMethod();
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
			case PooslPackage.DATA_METHOD_CALL__SUPER:
				setSuper((Boolean)newValue);
				return;
			case PooslPackage.DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				getObjectExpressions().clear();
				getObjectExpressions().addAll((Collection<? extends Expression>)newValue);
				return;
			case PooslPackage.DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
				getParameterExpressionLists().clear();
				getParameterExpressionLists().addAll((Collection<? extends ExpressionList>)newValue);
				return;
			case PooslPackage.DATA_METHOD_CALL__DATA_METHOD:
				setDataMethod((DataMethod)newValue);
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
			case PooslPackage.DATA_METHOD_CALL__SUPER:
				setSuper(SUPER_EDEFAULT);
				return;
			case PooslPackage.DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				getObjectExpressions().clear();
				return;
			case PooslPackage.DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
				getParameterExpressionLists().clear();
				return;
			case PooslPackage.DATA_METHOD_CALL__DATA_METHOD:
				setDataMethod((DataMethod)null);
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
			case PooslPackage.DATA_METHOD_CALL__SUPER:
				return super_ != SUPER_EDEFAULT;
			case PooslPackage.DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				return objectExpressions != null && !objectExpressions.isEmpty();
			case PooslPackage.DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
				return parameterExpressionLists != null && !parameterExpressionLists.isEmpty();
			case PooslPackage.DATA_METHOD_CALL__DATA_METHOD:
				return dataMethod != null;
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
		result.append(" (super: ");
		result.append(super_);
		result.append(')');
		return result.toString();
	}

} //DataMethodCallImpl
