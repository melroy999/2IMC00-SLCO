/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package poosl.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectResolvingEList;

import poosl.ExpressionList;
import poosl.PooslPackage;
import poosl.ProcessMethod;
import poosl.ProcessMethodCall;
import poosl.Variable;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Process Method Call</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.ProcessMethodCallImpl#getMethod <em>Method</em>}</li>
 *   <li>{@link poosl.impl.ProcessMethodCallImpl#getParameterExpressions <em>Parameter Expressions</em>}</li>
 *   <li>{@link poosl.impl.ProcessMethodCallImpl#getVariables <em>Variables</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ProcessMethodCallImpl extends StatementImpl implements ProcessMethodCall {
	/**
	 * The cached value of the '{@link #getMethod() <em>Method</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMethod()
	 * @generated
	 * @ordered
	 */
	protected ProcessMethod method;

	/**
	 * The cached value of the '{@link #getParameterExpressions() <em>Parameter Expressions</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getParameterExpressions()
	 * @generated
	 * @ordered
	 */
	protected EList<ExpressionList> parameterExpressions;

	/**
	 * The cached value of the '{@link #getVariables() <em>Variables</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getVariables()
	 * @generated
	 * @ordered
	 */
	protected EList<Variable> variables;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ProcessMethodCallImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.PROCESS_METHOD_CALL;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessMethod getMethod() {
		if (method != null && method.eIsProxy()) {
			InternalEObject oldMethod = (InternalEObject)method;
			method = (ProcessMethod)eResolveProxy(oldMethod);
			if (method != oldMethod) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PooslPackage.PROCESS_METHOD_CALL__METHOD, oldMethod, method));
			}
		}
		return method;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProcessMethod basicGetMethod() {
		return method;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMethod(ProcessMethod newMethod) {
		ProcessMethod oldMethod = method;
		method = newMethod;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.PROCESS_METHOD_CALL__METHOD, oldMethod, method));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ExpressionList> getParameterExpressions() {
		if (parameterExpressions == null) {
			parameterExpressions = new EObjectResolvingEList<ExpressionList>(ExpressionList.class, this, PooslPackage.PROCESS_METHOD_CALL__PARAMETER_EXPRESSIONS);
		}
		return parameterExpressions;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Variable> getVariables() {
		if (variables == null) {
			variables = new EObjectResolvingEList<Variable>(Variable.class, this, PooslPackage.PROCESS_METHOD_CALL__VARIABLES);
		}
		return variables;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PooslPackage.PROCESS_METHOD_CALL__METHOD:
				if (resolve) return getMethod();
				return basicGetMethod();
			case PooslPackage.PROCESS_METHOD_CALL__PARAMETER_EXPRESSIONS:
				return getParameterExpressions();
			case PooslPackage.PROCESS_METHOD_CALL__VARIABLES:
				return getVariables();
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
			case PooslPackage.PROCESS_METHOD_CALL__METHOD:
				setMethod((ProcessMethod)newValue);
				return;
			case PooslPackage.PROCESS_METHOD_CALL__PARAMETER_EXPRESSIONS:
				getParameterExpressions().clear();
				getParameterExpressions().addAll((Collection<? extends ExpressionList>)newValue);
				return;
			case PooslPackage.PROCESS_METHOD_CALL__VARIABLES:
				getVariables().clear();
				getVariables().addAll((Collection<? extends Variable>)newValue);
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
			case PooslPackage.PROCESS_METHOD_CALL__METHOD:
				setMethod((ProcessMethod)null);
				return;
			case PooslPackage.PROCESS_METHOD_CALL__PARAMETER_EXPRESSIONS:
				getParameterExpressions().clear();
				return;
			case PooslPackage.PROCESS_METHOD_CALL__VARIABLES:
				getVariables().clear();
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
			case PooslPackage.PROCESS_METHOD_CALL__METHOD:
				return method != null;
			case PooslPackage.PROCESS_METHOD_CALL__PARAMETER_EXPRESSIONS:
				return parameterExpressions != null && !parameterExpressions.isEmpty();
			case PooslPackage.PROCESS_METHOD_CALL__VARIABLES:
				return variables != null && !variables.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //ProcessMethodCallImpl
