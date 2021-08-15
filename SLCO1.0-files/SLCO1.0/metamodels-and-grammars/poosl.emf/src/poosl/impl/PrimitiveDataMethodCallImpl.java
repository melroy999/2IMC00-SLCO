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

import poosl.Expression;
import poosl.ExpressionList;
import poosl.PooslPackage;
import poosl.PrimitiveDataMethodCall;
import poosl.PrimitiveDataMethodEnum;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Primitive Data Method Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.PrimitiveDataMethodCallImpl#getObjectExpressions <em>Object Expressions</em>}</li>
 *   <li>{@link poosl.impl.PrimitiveDataMethodCallImpl#getParameterExpressionLists <em>Parameter Expression Lists</em>}</li>
 *   <li>{@link poosl.impl.PrimitiveDataMethodCallImpl#getPrimitiveDataMethod <em>Primitive Data Method</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class PrimitiveDataMethodCallImpl extends ExpressionImpl implements PrimitiveDataMethodCall {
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
	 * The default value of the '{@link #getPrimitiveDataMethod() <em>Primitive Data Method</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPrimitiveDataMethod()
	 * @generated
	 * @ordered
	 */
	protected static final PrimitiveDataMethodEnum PRIMITIVE_DATA_METHOD_EDEFAULT = PrimitiveDataMethodEnum.ADD;

	/**
	 * The cached value of the '{@link #getPrimitiveDataMethod() <em>Primitive Data Method</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPrimitiveDataMethod()
	 * @generated
	 * @ordered
	 */
	protected PrimitiveDataMethodEnum primitiveDataMethod = PRIMITIVE_DATA_METHOD_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected PrimitiveDataMethodCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.PRIMITIVE_DATA_METHOD_CALL;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Expression> getObjectExpressions() {
		if (objectExpressions == null) {
			objectExpressions = new EObjectContainmentEList<Expression>(Expression.class, this, PooslPackage.PRIMITIVE_DATA_METHOD_CALL__OBJECT_EXPRESSIONS);
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
			parameterExpressionLists = new EObjectContainmentEList<ExpressionList>(ExpressionList.class, this, PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS);
		}
		return parameterExpressionLists;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PrimitiveDataMethodEnum getPrimitiveDataMethod() {
		return primitiveDataMethod;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPrimitiveDataMethod(PrimitiveDataMethodEnum newPrimitiveDataMethod) {
		PrimitiveDataMethodEnum oldPrimitiveDataMethod = primitiveDataMethod;
		primitiveDataMethod = newPrimitiveDataMethod == null ? PRIMITIVE_DATA_METHOD_EDEFAULT : newPrimitiveDataMethod;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PRIMITIVE_DATA_METHOD, oldPrimitiveDataMethod, primitiveDataMethod));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				return ((InternalEList<?>)getObjectExpressions()).basicRemove(otherEnd, msgs);
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
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
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				return getObjectExpressions();
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
				return getParameterExpressionLists();
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PRIMITIVE_DATA_METHOD:
				return getPrimitiveDataMethod();
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
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				getObjectExpressions().clear();
				getObjectExpressions().addAll((Collection<? extends Expression>)newValue);
				return;
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
				getParameterExpressionLists().clear();
				getParameterExpressionLists().addAll((Collection<? extends ExpressionList>)newValue);
				return;
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PRIMITIVE_DATA_METHOD:
				setPrimitiveDataMethod((PrimitiveDataMethodEnum)newValue);
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
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				getObjectExpressions().clear();
				return;
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
				getParameterExpressionLists().clear();
				return;
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PRIMITIVE_DATA_METHOD:
				setPrimitiveDataMethod(PRIMITIVE_DATA_METHOD_EDEFAULT);
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
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__OBJECT_EXPRESSIONS:
				return objectExpressions != null && !objectExpressions.isEmpty();
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PARAMETER_EXPRESSION_LISTS:
				return parameterExpressionLists != null && !parameterExpressionLists.isEmpty();
			case PooslPackage.PRIMITIVE_DATA_METHOD_CALL__PRIMITIVE_DATA_METHOD:
				return primitiveDataMethod != PRIMITIVE_DATA_METHOD_EDEFAULT;
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
		result.append(" (primitiveDataMethod: ");
		result.append(primitiveDataMethod);
		result.append(')');
		return result.toString();
	}

} //PrimitiveDataMethodCallImpl
