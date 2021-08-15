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
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.EObjectResolvingEList;
import org.eclipse.emf.ecore.util.InternalEList;

import poosl.Parameter;
import poosl.PooslPackage;
import poosl.ProcessMethod;
import poosl.Statement;
import poosl.Variable;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Process Method</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link poosl.impl.ProcessMethodImpl#getInputParameters <em>Input Parameters</em>}</li>
 *   <li>{@link poosl.impl.ProcessMethodImpl#getOutputParameters <em>Output Parameters</em>}</li>
 *   <li>{@link poosl.impl.ProcessMethodImpl#getVariables <em>Variables</em>}</li>
 *   <li>{@link poosl.impl.ProcessMethodImpl#getName <em>Name</em>}</li>
 *   <li>{@link poosl.impl.ProcessMethodImpl#getBody <em>Body</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ProcessMethodImpl extends EObjectImpl implements ProcessMethod {
	/**
	 * The cached value of the '{@link #getInputParameters() <em>Input Parameters</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInputParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<Parameter> inputParameters;

	/**
	 * The cached value of the '{@link #getOutputParameters() <em>Output Parameters</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOutputParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<Parameter> outputParameters;

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
	 * The cached value of the '{@link #getBody() <em>Body</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getBody()
	 * @generated
	 * @ordered
	 */
	protected EList<Statement> body;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ProcessMethodImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PooslPackage.Literals.PROCESS_METHOD;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Parameter> getInputParameters() {
		if (inputParameters == null) {
			inputParameters = new EObjectResolvingEList<Parameter>(Parameter.class, this, PooslPackage.PROCESS_METHOD__INPUT_PARAMETERS);
		}
		return inputParameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Parameter> getOutputParameters() {
		if (outputParameters == null) {
			outputParameters = new EObjectResolvingEList<Parameter>(Parameter.class, this, PooslPackage.PROCESS_METHOD__OUTPUT_PARAMETERS);
		}
		return outputParameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Variable> getVariables() {
		if (variables == null) {
			variables = new EObjectContainmentEList<Variable>(Variable.class, this, PooslPackage.PROCESS_METHOD__VARIABLES);
		}
		return variables;
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
			eNotify(new ENotificationImpl(this, Notification.SET, PooslPackage.PROCESS_METHOD__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Statement> getBody() {
		if (body == null) {
			body = new EObjectContainmentEList<Statement>(Statement.class, this, PooslPackage.PROCESS_METHOD__BODY);
		}
		return body;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PooslPackage.PROCESS_METHOD__VARIABLES:
				return ((InternalEList<?>)getVariables()).basicRemove(otherEnd, msgs);
			case PooslPackage.PROCESS_METHOD__BODY:
				return ((InternalEList<?>)getBody()).basicRemove(otherEnd, msgs);
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
			case PooslPackage.PROCESS_METHOD__INPUT_PARAMETERS:
				return getInputParameters();
			case PooslPackage.PROCESS_METHOD__OUTPUT_PARAMETERS:
				return getOutputParameters();
			case PooslPackage.PROCESS_METHOD__VARIABLES:
				return getVariables();
			case PooslPackage.PROCESS_METHOD__NAME:
				return getName();
			case PooslPackage.PROCESS_METHOD__BODY:
				return getBody();
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
			case PooslPackage.PROCESS_METHOD__INPUT_PARAMETERS:
				getInputParameters().clear();
				getInputParameters().addAll((Collection<? extends Parameter>)newValue);
				return;
			case PooslPackage.PROCESS_METHOD__OUTPUT_PARAMETERS:
				getOutputParameters().clear();
				getOutputParameters().addAll((Collection<? extends Parameter>)newValue);
				return;
			case PooslPackage.PROCESS_METHOD__VARIABLES:
				getVariables().clear();
				getVariables().addAll((Collection<? extends Variable>)newValue);
				return;
			case PooslPackage.PROCESS_METHOD__NAME:
				setName((String)newValue);
				return;
			case PooslPackage.PROCESS_METHOD__BODY:
				getBody().clear();
				getBody().addAll((Collection<? extends Statement>)newValue);
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
			case PooslPackage.PROCESS_METHOD__INPUT_PARAMETERS:
				getInputParameters().clear();
				return;
			case PooslPackage.PROCESS_METHOD__OUTPUT_PARAMETERS:
				getOutputParameters().clear();
				return;
			case PooslPackage.PROCESS_METHOD__VARIABLES:
				getVariables().clear();
				return;
			case PooslPackage.PROCESS_METHOD__NAME:
				setName(NAME_EDEFAULT);
				return;
			case PooslPackage.PROCESS_METHOD__BODY:
				getBody().clear();
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
			case PooslPackage.PROCESS_METHOD__INPUT_PARAMETERS:
				return inputParameters != null && !inputParameters.isEmpty();
			case PooslPackage.PROCESS_METHOD__OUTPUT_PARAMETERS:
				return outputParameters != null && !outputParameters.isEmpty();
			case PooslPackage.PROCESS_METHOD__VARIABLES:
				return variables != null && !variables.isEmpty();
			case PooslPackage.PROCESS_METHOD__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case PooslPackage.PROCESS_METHOD__BODY:
				return body != null && !body.isEmpty();
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
		result.append(')');
		return result.toString();
	}

} //ProcessMethodImpl
