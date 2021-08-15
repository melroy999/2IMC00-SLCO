/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package nqc.impl;

import nqc.ConstantExpression;
import nqc.NqcPackage;

import org.eclipse.emf.ecore.EClass;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Constant Expression</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * </p>
 *
 * @generated
 */
public abstract class ConstantExpressionImpl extends ValueExpressionImpl implements ConstantExpression {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ConstantExpressionImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return NqcPackage.eINSTANCE.getConstantExpression();
	}

} //ConstantExpressionImpl
